#include <iostream>
#include <fstream>
#include <tclap/CmdLine.h> //-I wherever TCLAP is
#include <cuda.h>
#include <curand.h>
#define TILE_DIM 16

//TRY TO PUT A SIGMOID FUNCTOR HERE !!!!
__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, 
    int BRows, int BCols, int CRows, int CCols) 
{
    float CValue = 0;
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++)
    {
         if (k * TILE_DIM + threadIdx.x < ACols && Row < ARows)   
            As[threadIdx.y][threadIdx.x] = 
                A[Row * ACols + k * TILE_DIM + threadIdx.x];
         else                                                 
            As[threadIdx.y][threadIdx.x] = 0.0;

         if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols)   
            Bs[threadIdx.y][threadIdx.x] = 
                B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
         else      
            Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n) 
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols) 
        C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + 
            (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}

std::vector<int>& splitToInts(const std::string &s, char delim, 
                                std::vector<int> &elems);
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
    int i,j,l;
    float *X, *Y, *d_Y, **Theta, **d_Theta, **a, **d_a; 
    //d_Theta and d_a are host vectors of pointers to device memory!
    cudaError_t err;
	struct Options options = ParseCommandLine(argc,argv);
	
	printf("Number of layers: %d\n", options.numberOfLayers);
	
	printf("Activation function: %s\n", options.activationFunction.c_str());
	//==========================================================================
	// Allocate memory in both host and device
	//==========================================================================
	X = (float *) malloc (sizeof(float) * options.numberOfTrainingSamples * 
	                        options.layerSizes[0]);
	Y = (float *) malloc (sizeof(float) * options.numberOfTrainingSamples *
	                        options.layerSizes.back());
	Theta = (float **) malloc ((options.numberOfLayers - 1) * sizeof(float*)); //points to host memory
	d_Theta = (float **) malloc ((options.numberOfLayers - 1) * sizeof(float*)); // points to device memory
	a = (float **) malloc (options.numberOfLayers * sizeof(float*));
	d_a = (float **) malloc (options.numberOfLayers * sizeof(float*));
	                      
    /*cudaMalloc((void **) &d_X, options.numberOfTrainingSamples * 
                options.layerSizes[0] * sizeof(float));*/
    cudaMalloc((void **) &d_Y, options.numberOfTrainingSamples * 
                options.layerSizes.back() * sizeof(float));
	
	for (i = 0; i < options.numberOfLayers - 1; i++)
	{
	    Theta[i] =  (float *) malloc (sizeof(float) * options.layerSizes[i] * 
	                    options.layerSizes[i+1]);
	    if (Theta[i] == NULL)
	        printf ("MALLOC ERROR\n");                
        err = cudaMalloc((void **)&(d_Theta[i]), options.layerSizes[i] * 
                options.layerSizes[i+1] * sizeof(float));
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
    
    /*Let use Y as output for test for now
	readResultsIntoMatrix(options.resultsFile, Y, options.numberOfTrainingSamples, 
                            options.layerSizes.back());*/
    for (i = 0; i < options.numberOfLayers - 1; i++){
	   printf ("Filling Theta[%d] of size (%d,%d)\n",i, options.layerSizes[i],options.layerSizes[i+1]);
        GPU_fill_rand(d_Theta[i], options.layerSizes[i],
                        options.layerSizes[i+1]);
    }
	
	//==========================================================================
	// COMPUTE
	//==========================================================================
	printf ("Computing ----------------------------------------------------\n");
	//Feed the X to the first activation function: d_a[0]
	cudaMemcpy(d_a[0], X, options.numberOfTrainingSamples * 
	            options.layerSizes[0] * sizeof(float), cudaMemcpyHostToDevice);
	             
	dim3 dimBlock(TILE_DIM, TILE_DIM);
	// d_a[1] = d_a[0] x d_Theta[0]
	//multiplication scope, so I can reuse (redeclare) dimGrid ;)
	{ 
    dim3 dimGrid((options.numberOfTrainingSamples + dimBlock.x - 1) / dimBlock.x, 
                 (options.layerSizes[1] + dimBlock.y - 1)/ dimBlock.y);
    MatMul<<<dimGrid, dimBlock>>>(d_a[0], d_Theta[0], d_a[1],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[0],
                                    options.layerSizes[0],
                                    options.layerSizes[1],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[1]);
    cudaThreadSynchronize();
    }
    {
    dim3 dimGrid((options.numberOfTrainingSamples + dimBlock.x - 1) / dimBlock.x, 
                 (options.layerSizes[2] + dimBlock.y - 1)/ dimBlock.y);
    MatMul<<<dimGrid, dimBlock>>>(d_a[1], d_Theta[1], d_a[2],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[1],
                                    options.layerSizes[1],
                                    options.layerSizes[2],
                                    options.numberOfTrainingSamples,
                                    options.layerSizes[2]);
    cudaThreadSynchronize();
    }
    
    cudaMemcpy(Theta[0], d_Theta[0], options.layerSizes[0] * options.layerSizes[1] * sizeof(float), cudaMemcpyDeviceToHost);
     cudaMemcpy(Theta[1], d_Theta[1], options.layerSizes[1] * options.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(a[0], d_a[0], options.numberOfTrainingSamples * options.layerSizes[0] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(a[1], d_a[1], options.numberOfTrainingSamples * options.layerSizes[1] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(a[2], d_a[2], options.numberOfTrainingSamples * options.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
    
    printMatrix(Theta[0], options.layerSizes[0], options.layerSizes[1]);
    printMatrix(Theta[0], options.layerSizes[1], options.layerSizes[2]);
	printMatrix(a[0], options.numberOfTrainingSamples, options.layerSizes[0]);
	printMatrix(a[1], options.numberOfTrainingSamples, options.layerSizes[1]);
	printMatrix(a[2], options.numberOfTrainingSamples, options.layerSizes[2]);
	
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
			M[i * columns + x] = 1.0; 
	}
}

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    printf("Fill rand: (%d, %d)\n", nr_rows_A, nr_cols_A);
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

