#include <iostream>
#include <fstream>
#include <tclap/CmdLine.h> //-I wherever TCLAP is
#include <cuda.h>
#include <curand.h>
#include <math.h>

#define TILE_DIM 8
//========================================================================================
// SIGNATURES
//========================================================================================
struct sigmoidFunc {
        __host__ __device__ float operator()(float z) const {
        return 1.0/(1.0 + exp(-(z)));
    }
};

struct plusFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x + y;
    }
};

struct prodFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x * y;
    }
};

struct goodLogisticRegressionFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return (-x * log(y));
    }
};

struct badLogisticRegressionFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return ((1-x) * log(1-y));
    }
};

template<typename UnaryFunction>
__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, 
                       int BCols, int CRows, int CCols, bool addBias, 
                       UnaryFunction activationFunction);
template<typename MapFunction,
         typename ReduceFunction>                       
__global__ void ZipMapReduceKernel(float* X, float* Y, float* R, int size, 
                                    MapFunction mapFunction, float neutralElement, 
                                    ReduceFunction reduceFunction);
template<typename MapFunction,
         typename ReduceFunction>                                                           
float ZipMapReduce(float* d_X, float* d_Y, int size, MapFunction mapFunction, 
                   float neutralElement, ReduceFunction reduceFunction);
                   
std::vector<int>& splitToInts(const std::string &s, char delim, std::vector<int> &elems);
std::vector<int> splitToInts(const std::string &s, char delim);

struct Options 
{
    int numberOfLayers;
	std::vector<int> layerSizes;
	std::string activationFunction;
	std::string samplesFile;
	std::string resultsFile;
	int numberOfTrainingSamples;
};
struct Options ParseCommandLine(int argc, char *argv[]);
void readCsvIntoMatrix(const std::string fileName, float* M, const int rows, 
                        const int columns);
void readResultsIntoMatrix(const std::string fileName, float* M, const int rows, 
                            const int columns);
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A);
void printMatrix(float *M, int rows, int columns);

int main(int argc, char *argv[])
{
    int i,l;
    float *X, *Y, *d_Y, **Theta, **d_Theta, **a, **d_a, J; 
    //d_Theta and d_a are host vectors of pointers to device memory!
    cudaError_t err;
    sigmoidFunc sigmoidf;
    goodLogisticRegressionFunc goodLogisticRegressionf;
    badLogisticRegressionFunc badLogisticRegressionf;
    plusFunc plusf;
    prodFunc prodf;
    
	Options options = ParseCommandLine(argc,argv);
	
	printf("Number of layers: %d\n", options.numberOfLayers);
	
	printf("Activation function: %s\n", options.activationFunction.c_str());
	//==========================================================================
	// Allocate memory in both host and device
	//==========================================================================
	X = (float *) malloc (sizeof(float) * options.numberOfTrainingSamples * 
	                        options.layerSizes[0]);
	Y = (float *) malloc (sizeof(float) * options.numberOfTrainingSamples *
	                        options.layerSizes.back());
	Theta = (float **) malloc ((options.numberOfLayers - 1) * sizeof(float*)); 
	d_Theta = (float **) malloc ((options.numberOfLayers - 1) * sizeof(float*));
	a = (float **) malloc (options.numberOfLayers * sizeof(float*));
	d_a = (float **) malloc (options.numberOfLayers * sizeof(float*));
	
    cudaMalloc((void **) &d_Y, options.numberOfTrainingSamples * 
                options.layerSizes.back() * sizeof(float));
	for (i = 0; i < options.numberOfLayers - 1; i++)
	{
	    Theta[i] =  (float *) malloc (sizeof(float) * (options.layerSizes[i] + 1) * 
	                    options.layerSizes[i+1]); //+1 is the bias row
	    if (Theta[i] == NULL)
	        printf ("MALLOC ERROR\n");                
        err = cudaMalloc((void **)&(d_Theta[i]), (options.layerSizes[i] + 1) * 
                options.layerSizes[i+1] * sizeof(float)); //+1 is the bias row
        if (err > 0) printf("error code: %d\n",err);
    }
    
    for (l = 0; l < options.numberOfLayers; l++) 
    {
        //an activation for each training sample per each neuron at layer l
        err = cudaMalloc((void **)&(d_a[l]), options.layerSizes[l] * 
                options.numberOfTrainingSamples * sizeof(float)); 
        a[l] = (float *) malloc(sizeof(float) * options.layerSizes[l] * 
                options.numberOfTrainingSamples); 
        if (err > 0) printf("error code: %d\n",err);
        
    }
    
    
	//==========================================================================
	// INITIALIZE VALUES
	//==========================================================================
	readCsvIntoMatrix(options.samplesFile, X, options.numberOfTrainingSamples, 
                        options.layerSizes[0]);
	readResultsIntoMatrix(options.resultsFile, Y, options.numberOfTrainingSamples, 
                            options.layerSizes.back());
    //for (i = 0; i < options.numberOfLayers - 1; i++)
    //    GPU_fill_rand(d_Theta[i], options.layerSizes[i] + 1, options.layerSizes[i+1]);
	readCsvIntoMatrix("data_theta0.csv", Theta[0], options.layerSizes[0] + 1, 
                        options.layerSizes[1]);
	readCsvIntoMatrix("data_theta1.csv", Theta[1], options.layerSizes[1] + 1, 
                        options.layerSizes[2]);
    cudaMemcpy(d_Theta[0], Theta[0], (options.layerSizes[0] + 1) * options.layerSizes[1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Theta[1], Theta[1], (options.layerSizes[1] + 1) * options.layerSizes[2] * sizeof(float), cudaMemcpyHostToDevice);
                        
	//==========================================================================
	// COMPUTE
	//==========================================================================
	printf ("Computing ----------------------------------------------------\n");
	//Feed the X to the first activation function: d_a[0]
	//Fedd the Y
	cudaMemcpy(d_a[0], X, options.numberOfTrainingSamples * 
	            options.layerSizes[0] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, options.numberOfTrainingSamples * 
	           options.layerSizes.back() * sizeof(float), cudaMemcpyHostToDevice);
	           
	dim3 dimBlock(TILE_DIM, TILE_DIM);
	// d_a[1] = d_a[0] x d_Theta[0]
	//multiplication scope, so I can reuse (redeclare) dimGrid ;)
	{ 
    dim3 dimGrid((options.layerSizes[1] + dimBlock.x - 1)/ dimBlock.x, 
                 (options.numberOfTrainingSamples + dimBlock.y - 1)/ dimBlock.y);
                 
    MatMul<<<dimGrid, dimBlock>>>(d_a[0], d_Theta[0], d_a[1],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[0],
                                    options.layerSizes[0],
                                    options.layerSizes[1],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[1],
                                    true, sigmoidf);
    cudaThreadSynchronize();
    }
    {
    dim3 dimGrid((options.layerSizes[2] + dimBlock.x - 1) / dimBlock.x, 
                 (options.numberOfTrainingSamples + dimBlock.y - 1)/ dimBlock.y);
                 
    MatMul<<<dimGrid, dimBlock>>>(d_a[1], d_Theta[1], d_a[2],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[1],
                                    options.layerSizes[1],
                                    options.layerSizes[2],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[2],
                                    true, sigmoidf);
    cudaThreadSynchronize();
    }
    
	
	
    //cudaMemcpy(Theta[0], d_Theta[0], (options.layerSizes[0] + 1) * options.layerSizes[1] * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Theta[1], d_Theta[1], (options.layerSizes[1] + 1) * options.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(a[0], d_a[0], options.numberOfTrainingSamples * options.layerSizes[0] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(a[1], d_a[1], options.numberOfTrainingSamples * options.layerSizes[1] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(a[2], d_a[2], options.numberOfTrainingSamples * options.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
    
	//printMatrix(a[0], options.numberOfTrainingSamples, options.layerSizes[0]);
	//printMatrix(Theta[0], options.layerSizes[0]+1, options.layerSizes[1]);
	//printMatrix(a[1], options.numberOfTrainingSamples, options.layerSizes[1]);
	//printMatrix(Theta[1], options.layerSizes[1]+1, options.layerSizes[2]);
	//printMatrix(a[2], options.numberOfTrainingSamples, options.layerSizes[2]);
	//printMatrix(Y,options.numberOfTrainingSamples,options.layerSizes.back());
	
	//Cost
	J = ZipMapReduce(d_Y, d_a[options.numberOfLayers - 1], 
	                 options.numberOfTrainingSamples * options.layerSizes.back(),        
	                 goodLogisticRegressionf, 0.0, plusf) -
	    ZipMapReduce(d_Y, d_a[options.numberOfLayers - 1], 
	                 options.numberOfTrainingSamples * options.layerSizes.back(), 
	                 badLogisticRegressionf,0.0,plusf);
	              
	J = J / options.numberOfTrainingSamples; //Average
	
	printf("Cost: %f\n",J); 
    
    //Regularized cost
	float coef = 0.0;
	for (i = 0; i < options.numberOfLayers - 1; i++)
	    coef += ZipMapReduce(d_Theta[i]+options.layerSizes[i+1], 
	                         d_Theta[i]+options.layerSizes[i+1],
	                          options.layerSizes[i] * options.layerSizes[i+1],                
	                          prodf,0.0,plusf);
	                          
	J += ( coef /(2*options.numberOfTrainingSamples));
	printf("Coef: %f\n",coef);
	printf("Regularized cost: %f\n",J); 
	
	
	//    cudaFree(d_X); cudaFree(d_B); cudaFree(d_C);    
	return 0;
}


///
/// Helper Functions
///
struct Options ParseCommandLine(int argc, char *argv[])
{
    struct Options options;
    TCLAP::CmdLine cmd("Command description message", ' ', "0.9");
	
	TCLAP::ValueArg<int> numLayersArg("L", "number-of-layers", 
	    "Number of layers considering the input and output layers", false, 3, 
	    "integer", cmd);
	    
	TCLAP::ValueArg<int> numTrainingSamplesArg("T", "number-training-samples", 
	    "Number of training samples to use", true, 0, "integer", cmd);
	    
	TCLAP::ValueArg<std::string> layersArg("l", "layer-sizes", 
	    "Number of neurons for each layer", true, "", "list of integers",cmd);
	
	std::vector<std::string> allowedActivationFunctions;
		allowedActivationFunctions.push_back("sigmoid");
		allowedActivationFunctions.push_back("htan");
	TCLAP::ValuesConstraint<std::string> allowedValsActivationFunction( 
	    allowedActivationFunctions );
	
	TCLAP::ValueArg<std::string> activationFunctionArg("a",
	    "activation-function", "Activation function", false, "sigmoid", 
	    &allowedValsActivationFunction, cmd);
	    
	TCLAP::ValueArg<std::string> fileXArg ("x", "samples", 
	    "File containing the training examples", true, "", 
	    "file name or path", cmd);
	    
    TCLAP::ValueArg<std::string> fileYArg ("y", "results", 
	    "File containing the training results", true, "", 
	    "file name or path", cmd);
	    
	cmd.parse( argc, argv );
	    options.numberOfLayers          = numLayersArg.getValue();
	    options.layerSizes              = splitToInts(layersArg.getValue(),',');
	    options.activationFunction      = activationFunctionArg.getValue();
	    options.samplesFile             = fileXArg.getValue();
	    options.resultsFile             = fileYArg.getValue();
        options.numberOfTrainingSamples = numTrainingSamplesArg.getValue();
        
    return options;
}

std::vector<int> &splitToInts(const std::string &s, char delim, 
    std::vector<int> &elems) 
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(atoi(item.c_str()));
    }
    return elems;
}

std::vector<int> splitToInts(const std::string &s, char delim) 
{
    std::vector<int> elems;
    splitToInts(s, delim, elems);
    return elems;
}

void readCsvIntoMatrix(const std::string fileName, float* M, const int rows, 
                        const int columns)
{
    std::ifstream ifs (fileName.c_str());
	char dummy;
	float x;
	
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < columns; ++j){
			ifs >> x;
			M[i * columns + j] = x; 
			if (j < (columns - 1)) //ignore commas
				ifs >> dummy;
		}
	}
}

void readResultsIntoMatrix(const std::string fileName, float* M, const int rows, 
                            const int columns)
{
    std::ifstream ifs (fileName.c_str());
	int x;
	
	for (int i = 0; i < rows; ++i){
			ifs >> x;
			for (int j = 0; j < columns; j++)
			    M[i * columns + j] = 0.0;
			M[i * columns + x - 1] = 1.0; 
	}
}

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    printf("Fill rand: (%p,%d, %d)\n", A, nr_rows_A, nr_cols_A);
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void printMatrix(float *M, int rows, int columns)
{
    printf("M:\n");
    for (int i = 0; i < rows; i++){
	    for (int j = 0; j < columns; j++)
	        printf("%f, ", M[i * columns + j]);
	    printf("\n");
	} 
}
//==============================================================================
// KERNELS WRAPPER FUNCTIONS
//==============================================================================
template<typename MapFunction,
         typename ReduceFunction>
float ZipMapReduce(float* d_X, float* d_Y, int size, MapFunction mapFunction, 
                   float neutralElement, ReduceFunction reduceFunction)
{
    float *R, *d_R, r = neutralElement;
    dim3 dimBlock(TILE_DIM);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);
    
    //Create auxiliary vector
    R = (float *) malloc (sizeof(float) * dimGrid.x);
    cudaError_t err = cudaMalloc((void **)&d_R, sizeof(float) * dimGrid.x);

    //Reduce to vector R
	ZipMapReduceKernel<<<dimGrid, dimBlock>>>(d_X, d_Y, d_R, size, mapFunction, neutralElement, reduceFunction);
    cudaThreadSynchronize();
    cudaMemcpy(R, d_R, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    /*printf("Reduced to %d values: \n", dimGrid.x);
    for (int i = 0; i < dimGrid.x; i++)
    {
        printf("%f, ", R[i]);
    }
    printf("\n");*/
    //Reduce remaining values in host
    for (int i = 0; i < dimGrid.x; i++)
        r = reduceFunction(r, R[i]);
    cudaFree(d_R);
    free(R);
    return r;
}
//==============================================================================
// KERNELS
//==============================================================================
template<typename UnaryFunction>
__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, 
    int BRows, int BCols, int CRows, int CCols, bool addBias, UnaryFunction activationFunction ) 
{
    float CValue = 0;
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;
    int biasOffset = addBias ? 1 : 0; 
	
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++)           //floor(ACols/TILE_DIM)
    {
         if (k * TILE_DIM + threadIdx.x < ACols && Row < ARows)   
            As[threadIdx.y][threadIdx.x] = A[Row * ACols + k * TILE_DIM + threadIdx.x];
         else                                                 
            As[threadIdx.y][threadIdx.x] = 0.0;

         if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols)   
            Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y + biasOffset) * BCols + Col]; //+1 one row if bias
         else      
            Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n) 
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }
    if (addBias)
	{
		__shared__ float BiasRow[TILE_DIM];
	    
		if (threadIdx.y == 0){
		  if (Col < BCols){
			BiasRow[threadIdx.x] = B[Col];
		  }else{
		  	BiasRow[threadIdx.x] = 0.0;
			}
			}
	    __syncthreads();
		
		CValue += BiasRow[threadIdx.x];
		
		__syncthreads();
		
	}
	
    if (Row < CRows && Col < CCols) 
        C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + 
          (blockIdx.x * blockDim.x) + threadIdx.x] =
            activationFunction(CValue);
}

template<typename MapFunction,
         typename ReduceFunction>
__global__ void ZipMapReduceKernel(float* X, float* Y, float* R, int size, 
                                    MapFunction mapFunction, float neutralElement, 
                                    ReduceFunction reduceFunction)
{
    __shared__ float sX[TILE_DIM];
    __shared__ float sY[TILE_DIM];
    __shared__ float sR[TILE_DIM];
  
    unsigned int i = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
        
    //Load data from memory to shared memory collectively
    sX[tid] = X[i];
    sY[tid] = Y[i];
    sR[tid] = neutralElement;
    __syncthreads();
    
    //Zip and Map: sR <- Map(Zip(sX,sY))
    if (i < size) 
        sR[tid] = mapFunction(sX[tid],sY[tid]);
    __syncthreads();
    
    //Reduce
    for(unsigned int s = TILE_DIM / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sR[tid] = reduceFunction(sR[tid], sR[tid + s]);
        __syncthreads();
    }
    
    //Delegate (thread 0) writes to memory
    if (tid == 0)
        R[bid] = sR[0];
}
