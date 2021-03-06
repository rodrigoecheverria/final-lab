#include <iostream>
#include <fstream>
#include <tclap/CmdLine.h> //-I wherever TCLAP is
#include <cuda.h>
#include <curand.h>
#include <math.h>

#define EPSILON 0.00001
#define TILE_DIM 32
//=============================================================================
// FUNCTORS
//=============================================================================
struct sigmoidFunc {
        __host__ __device__ float operator()(float z) const {
        return 1.0/(1.0 + exp(-(z)));
    }
};

struct dummyFunc {
        __host__ __device__ float operator()(float z) const {
        return 0.0;
    }
};

struct plusFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x + y;
    }
};

struct subFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x - y;
    }
};

struct prodFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x * y;
    }
};

struct sxpayFunc {
        float p;
		sxpayFunc(float _p) : p(_p) {}
        __host__ __device__ float operator()(float x, float y) const {
        return x + p*y;
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

struct gradientFunc {
        __host__ __device__ float operator()(float x, float y) const {
        sigmoidFunc sigmoidf;
        if (x > 0.99999999 && x < 1.00000001) // x == 1
            return (x);
        else
            return (sigmoidf(x) * (1-sigmoidf(y)));
    }
};

struct DivideByFunc {
        float d;
		DivideByFunc(float _d) : d(_d) {}
        __host__ __device__ float operator()(float x, float y) const {
        return (x/d);
    }
};

//=============================================================================
// SIGNATURES
//=============================================================================

void MatMul(float* A, float* B, float* C, int ARows, int ACols,  int BRows, 
            int BCols);

template<typename UnaryFunction>
void CalculateActivation(float* A, float* B, float* C, float* aC, int ARows, 
                         int ACols, int BRows, int BCols, bool addBias,
                         UnaryFunction activationFunction);

template<typename UnaryFunction>
__global__ void MatMulKernel(float* A, float* B, float* C, float* aC, int ARows, 
                             int ACols, int BRows, int BCols, int CRows, 
                             int CCols, bool addBias, 
                             UnaryFunction activationFunction );
                       
template<typename MapFunction,
         typename ReduceFunction>                       
__global__ void ZipMapReduceKernel(float* X, float* Y, float* R, int size, 
                                   MapFunction mapFunction, 
                                   float neutralElement, 
                                   ReduceFunction reduceFunction);
								   
template<typename MapFunction,
              typename ReduceFunction>                                                           
float ZipMapReduce(float* d_X, float* d_Y, int size, MapFunction mapFunction, 
                   float neutralElement, ReduceFunction reduceFunction);
 
template<typename MapFunction>                   
__global__ void ZipMapKernel(float* X, float* Y, float* R, int size, 
                             MapFunction mapFunction);
                             
template<typename MapFunction>                             
void ZipMap(float* d_X, float* d_Y, float* d_R, int size, 
            MapFunction mapFunction); 

__global__ void TransposeKernel(float *d_A, float *d_At, int rows, int cols);

void Transpose(float* d_A, float* d_B, int rows, int cols); 
              
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
	int maxIterations;
	float alpha;
};

struct Options ParseCommandLine(int argc, char *argv[]);

void readCsvIntoMatrix(const std::string fileName, float* M, const int rows, 
                       const int columns, bool addBiasColumn = false);
								  
void readResultsIntoMatrix(const std::string fileName, float* M, const int rows, 
                                       const int columns);
									   
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A);

void printMatrix(float *M, int rows, int columns);

void printMatrixFromDevice(float *M, int rows, int columns);

//=============================================================================
// MAIN
//=============================================================================
int main(int argc, char *argv[])
{
    int i,l,j;
	int lastDelta;
    float *X, *Y, *d_Y, **Theta, **d_Theta, **d_Theta_trans, 
          **d_z, **d_a, **d_a_trans, **d_delta, 
          **d_Delta, J, J2, diff; 
    //d_Theta and d_a are host vectors of pointers to device memory!
    
    cudaError_t err;
    sigmoidFunc sigmoidf;
    goodLogisticRegressionFunc goodLogisticRegressionf;
    badLogisticRegressionFunc badLogisticRegressionf;
    gradientFunc gradientf;
    plusFunc plusf;
    prodFunc prodf;
    subFunc  subf;

    
	Options o = ParseCommandLine(argc,argv);
	printf("Number of layers: %d\n", o.numberOfLayers);
	
	printf("Activation function: %s\n", o.activationFunction.c_str());
	
	DivideByFunc divideByNumberOfTrainingSamplesf(o.numberOfTrainingSamples);
	//=========================================================================
	// Allocate memory in both host and device
	//=========================================================================
	X = (float *) malloc (sizeof(float) * o.numberOfTrainingSamples * 
	                      (o.layerSizes[0] + 1));
	Y = (float *) malloc (sizeof(float) * o.numberOfTrainingSamples * 
	                      o.layerSizes.back());
	cudaMalloc((void **) &d_Y, o.numberOfTrainingSamples * o.layerSizes.back() * 
               sizeof(float));
               
    Theta         = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));          
	d_Theta       = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_Theta_trans = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_delta       = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_Delta       = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_z           = (float **) malloc (o.numberOfLayers * sizeof(float*));
	d_a           = (float **) malloc (o.numberOfLayers * sizeof(float*));
	d_a_trans     = (float **) malloc (o.numberOfLayers * sizeof(float*));
   
	for (i = 0; i < o.numberOfLayers - 1; i++)
	{      
	    Theta[i] = (float *) malloc (sizeof(float) * (o.layerSizes[i] + 1) * 
                         o.layerSizes[i+1]);
        err = cudaMalloc((void **)&(d_Theta[i]), (o.layerSizes[i] + 1) * 
                         o.layerSizes[i+1] * sizeof(float)); 
	    err = cudaMalloc((void **)&(d_Theta_trans[i]), 
	                     o.numberOfTrainingSamples * 
	                     o.layerSizes[i+1] * sizeof(float));
	    err = cudaMalloc((void **)&(d_Delta[i]), (o.layerSizes[i] + 1) * 
                         o.layerSizes[i + 1] * sizeof(float));         
    }
    //Check the indexes of Delta[1] : it should be right now
	 
	lastDelta = o.numberOfLayers - 2;
    err = cudaMalloc((void **)&(d_delta[lastDelta]), o.numberOfTrainingSamples * 
	                     o.layerSizes.back() * sizeof(float));
						 
	for (i = 0; i < lastDelta; i++)
	{
		err = cudaMalloc((void **)&(d_delta[i]), o.numberOfTrainingSamples * 
							 (o.layerSizes[i+1] + 1) * sizeof(float)); 
	}
	
    for (l = 0; l < o.numberOfLayers; l++) 
    {
        int spaceForBias = (l == o.numberOfLayers - 1) ? 0 : 1;
        //linear comb for each neuron
        err = cudaMalloc((void **)&(d_z[l]), o.layerSizes[l] * 
                o.numberOfTrainingSamples * sizeof(float)); 
        if (err > 0) printf("error code: %d\n",err);
        
        //an activation for each training sample per each neuron at layer l
        err = cudaMalloc((void **)&(d_a[l]), (o.layerSizes[l] + spaceForBias) * 
                o.numberOfTrainingSamples * sizeof(float));  
        if (err > 0) printf("error code: %d\n",err);
        
        //an activation (transposed) for each training sample per each neuron at 
        //layer l
        err = cudaMalloc((void **)&(d_a_trans[l]), (o.layerSizes[l] + spaceForBias) * 
                o.numberOfTrainingSamples * sizeof(float));  
        if (err > 0) printf("error code: %d\n",err);
        
    }
    
	//==========================================================================
	// INITIALIZE VALUES
	//==========================================================================
	readCsvIntoMatrix(o.samplesFile, X, o.numberOfTrainingSamples, 
                        o.layerSizes[0],true);
	readResultsIntoMatrix(o.resultsFile, Y, o.numberOfTrainingSamples, 
                            o.layerSizes.back());
    printf("Training set read\n");
    //for (i = 0; i < o.numberOfLayers - 1; i++)
    //    GPU_fill_rand(d_Theta[i], o.layerSizes[i] + 1, o.layerSizes[i+1]);
	
	//Force values 
	readCsvIntoMatrix("data_theta0.csv", Theta[0], o.layerSizes[0] + 1, 
                        o.layerSizes[1]);
	readCsvIntoMatrix("data_theta1.csv", Theta[1], o.layerSizes[1] + 1, 
                        o.layerSizes[2]);
    cudaMemcpy(d_Theta[0], Theta[0], (o.layerSizes[0] + 1) * o.layerSizes[1] * 
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Theta[1], Theta[1], (o.layerSizes[1] + 1) * o.layerSizes[2] * 
               sizeof(float), cudaMemcpyHostToDevice);
    printf("Theta read\n");
                        
	//==========================================================================
	// COMPUTE
	//==========================================================================
	printf ("Computing ----------------------------------------------------\n");
	//Feed the X to the first activation function: d_a[0]
	cudaMemcpy(d_a[0], X, o.numberOfTrainingSamples * (o.layerSizes[0] + 1) * 
	                   sizeof(float), cudaMemcpyHostToDevice);
	//Feed the Y
	cudaMemcpy(d_Y, Y, o.numberOfTrainingSamples * o.layerSizes.back() * 
	                   sizeof(float), cudaMemcpyHostToDevice);
    
	for(i = 0; i < o.numberOfLayers - 1; i++)
	{ 
	    bool addBias = (i+1 != o.numberOfLayers - 1);
		CalculateActivation(d_a[i], d_Theta[i], d_z[i+1], d_a[i+1],
                            o.numberOfTrainingSamples,
                            o.layerSizes[i] + 1,
                            o.layerSizes[i] + 1,
                            o.layerSizes[i+1],
                            addBias,
                            sigmoidf);
    }
	
	
	//Cost
	J = ZipMapReduce(d_Y, d_a[o.numberOfLayers - 1], 
					 o.numberOfTrainingSamples * o.layerSizes.back(),        
	                 goodLogisticRegressionf, 0.0, plusf) -
	    ZipMapReduce(d_Y, d_a[o.numberOfLayers - 1], 
	                 o.numberOfTrainingSamples * o.layerSizes.back(), 
	                 badLogisticRegressionf,0.0,plusf);
	              
	J = J / o.numberOfTrainingSamples; //Average
	
	printf("Cost: %f\n",J); 
    
    //Regularized cost 
	float coef = 0.0;
	for (i = 0; i < o.numberOfLayers - 1; i++)
	    coef += ZipMapReduce(d_Theta[i]+o.layerSizes[i+1], 
	                         d_Theta[i]+o.layerSizes[i+1],
	                         o.layerSizes[i] * o.layerSizes[i+1],                
	                         prodf, 0.0, plusf);
	                          
	J += ( coef /(2*o.numberOfTrainingSamples));
	printf("Coef: %f\n",coef);
	printf("Regularized cost: %f\n",J); 
	
    /*
	 * BACKPROPAGATION
	 
	j = 0;
	do {
	int lastDelta = o.numberOfLayers - 2;
	
	//Calculate the error in the output: delta[lastDelta] <- d_a[last] - Y
	ZipMap(d_a[o.numberOfLayers - 1], d_Y, d_delta[lastDelta], 
	           o.numberOfTrainingSamples * o.layerSizes.back(), subf);
	
    //Backpropagate the error up to the first hidden layer (the first layer has 
    //no error, it is the input)
    //d_delta[l] <- d_delta[l+1] x d_Theta_trans[l+1]) .* g'(z[l+1])
    //Where z is the linear combination at layer l (without activation) and g' 
    //is the derivative of the activation function
    for (i = lastDelta - 1; i >= 0; i--)
    {    
		Transpose(d_Theta[i + 1], d_Theta_trans[i + 1], o.layerSizes[i + 1] + 1, 
		          o.layerSizes[i + 2]);
		
		//First term (d_delta[l+1] x d_Theta_trans[l+1]):
		MatMul (d_delta[i + 1], d_Theta_trans[i + 1], d_delta[i],
					o.numberOfTrainingSamples,
					o.layerSizes[i + 2],
					o.layerSizes[i + 2],
					o.layerSizes[i + 1] + 1);    
					
		//Second term (g'(z[l+1]))
		ZipMap(d_z[i + 1], d_z[i + 1], d_z[i + 1], o.numberOfTrainingSamples * 
		       (o.layerSizes[i + 1] + 1), gradientf);
		
		//Element wise product 
		ZipMap(d_delta[i], d_z[i + 1], d_delta[i], o.numberOfTrainingSamples * 
		      (o.layerSizes[i + 1] + 1), prodf);
	}    
    //UP TO THIS POINT EVERYTHING IS FINE

	//Calculate the Delta (gradient to be applied in Theta_i to correct it)
	//Delta[l] <- delta[l] x a[l]
	
	Transpose(d_delta[lastDelta], d_delta[lastDelta], o.numberOfTrainingSamples,
	          o.layerSizes.back());
                          
    MatMul (d_delta[lastDelta], d_a[lastDelta], d_Delta[lastDelta],
	            o.layerSizes[lastDelta + 1],
				o.numberOfTrainingSamples,
				o.numberOfTrainingSamples,
				o.layerSizes[lastDelta]);
                          
	Transpose(d_Delta[lastDelta], d_Delta[lastDelta], 
	          o.layerSizes[lastDelta + 1], o.layerSizes[lastDelta]);
	
	//Divide all elements in delta by the number of samples                          
	ZipMap(d_Delta[lastDelta], d_Delta[lastDelta], d_Delta[lastDelta], 
	           o.layerSizes[lastDelta + 1] * o.layerSizes[lastDelta],           
	           divideByNumberOfTrainingSamplesf);
	//CHECK'D                          
	for (i = lastDelta - 1; i >= 0; i--)
    {
		Transpose(d_delta[i], d_delta[i], o.numberOfTrainingSamples, 
		          o.layerSizes[i + 1] + 1);
                              
		MatMul (d_delta[i] + o.numberOfTrainingSamples, d_a[i], d_Delta[i],
					o.layerSizes[i + 1],
					o.numberOfTrainingSamples,
					o.numberOfTrainingSamples,
					o.layerSizes[i]);		
					
		Transpose(d_Delta[i], d_Delta[i], o.layerSizes[i + 1] , 
		          o.layerSizes[i]);
		                 
		//Divide all elements in delta by the number of samples
		ZipMap(d_Delta[i], d_Delta[i], d_Delta[i], 
	               o.layerSizes[i + 1] * (o.layerSizes[i]+1), 
	               divideByNumberOfTrainingSamplesf);

	}
	
	float rescue = 0.001;
	for (i = 0; i <= 26; i++)
        cudaMemcpy(d_Delta[1] + (o.layerSizes[2] * (o.layerSizes[1] + 1) - i), 
        &rescue, sizeof(float), cudaMemcpyHostToDevice);
    for (i = 0; i <=401; i++)
        cudaMemcpy(d_Delta[0] + (o.layerSizes[1] * (o.layerSizes[0] + 1) - i), 
        &rescue, sizeof(float), cudaMemcpyHostToDevice);
	//printMatrixFromDevice(d_Delta[1], o.layerSizes[2] , 
    //                          o.layerSizes[1] + 1);
    //printMatrixFromDevice(d_Delta[0], o.layerSizes[1] , 
    //                          o.layerSizes[0] + 1);		  
	//Regularize Deltas
	for (i = 0; i <= lastDelta; i++)
	{
		ZipMap(d_Delta[i], d_Theta[i], d_Delta[i], o.layerSizes[i + 1]  *
		       (o.layerSizes[i] + 1), sxpayFunc(1/o.numberOfTrainingSamples)); //REVISE	
	}
	
	//Apply Deltas
	for (i = 0; i <= lastDelta; i++)
	{
		ZipMap(d_Theta[i], d_Delta[i], d_Theta[i], o.layerSizes[i + 1] * 
		       (o.layerSizes[i] + 1), plusf);
		//printMatrixFromDevice(d_Theta[i], o.layerSizes[i + 1] , 
        //                      o.layerSizes[i] + 1);
	}
	//=========================================================================
	//Recalculate cost
	//=========================================================================
	for(i = 0; i < o.numberOfLayers - 1; i++)
	{ 
		CalculateActivation(d_a[i], d_Theta[i], d_z[i+1], d_a[i+1],
                                     o.numberOfTrainingSamples,
                                     o.layerSizes[i],
                                     o.layerSizes[i],
                                     o.layerSizes[i+1],
                                     sigmoidf);
    }
	
	//Cost
	J2 = ZipMapReduce(d_Y, d_a[o.numberOfLayers - 1], 
					           o.numberOfTrainingSamples * o.layerSizes.back(),        
	                           goodLogisticRegressionf, 0.0, plusf) -
	     ZipMapReduce(d_Y, d_a[o.numberOfLayers - 1], 
	                           o.numberOfTrainingSamples * o.layerSizes.back(), 
	                           badLogisticRegressionf,0.0,plusf);
	              
	J2 = J2 / o.numberOfTrainingSamples; //Average
	    
    //Regularized cost 
	float coef = 0.0;
	for (i = 0; i < o.numberOfLayers - 1; i++)
	    coef += ZipMapReduce(d_Theta[i]+o.layerSizes[i+1], 
	                         d_Theta[i]+o.layerSizes[i+1],
	                         o.layerSizes[i] * o.layerSizes[i+1],                
	                         prodf, 0.0, sxpayFunc(o.alpha));
	J2 += ( coef /(2*o.numberOfTrainingSamples));
	printf("Iteration %d: Regularized cost(coef: %f): %f\n", j,coef, J2); 
	
	diff = J - J2;
	
	J = J2;
	j++;
	} while (diff > EPSILON || j <= o.maxIterations);*/
	
	
    //==========================================================================
	// FREE MEMORY
	//==========================================================================     
	cudaFree(d_Y);
	for (i = 0; i < o.numberOfLayers - 1; i++) cudaFree(d_Theta[i]);
	for (i = 0; i < o.numberOfLayers - 1; i++) cudaFree(d_Theta_trans[i]);
	for (i = 0; i < o.numberOfLayers - 1; i++) cudaFree(d_delta[i]);
	for (i = 0; i < o.numberOfLayers - 1; i++) cudaFree(d_Delta[i]);
	for (i = 0; i < o.numberOfLayers; i++) cudaFree(d_a[i]);    
	for (i = 0; i < o.numberOfLayers; i++) cudaFree(d_z[i]);  
	 
	free(X);
	free(Y);
	free(d_Theta);
	free(d_a);
	free(d_z);
	free(d_delta);
	free(d_Delta);

	return 0;
}


///
/// Helper Functions
///
struct Options ParseCommandLine(int argc, char *argv[])
{
    struct Options o;
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
		//allowedActivationFunctions.push_back("htan"); 
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
    TCLAP::ValueArg<int> maxIterArg ("i","max-iterations", "Maximum number of \
	 iterations for the backpropagation",  false, 50, "integer", cmd);
	TCLAP::ValueArg<float> alphaArg ("r","learning-rate", "Learning rate for \
	the backpropagation",  false, 0.50, "float", cmd);
	cmd.parse( argc, argv );
	    o.numberOfLayers                = numLayersArg.getValue();
	    o.layerSizes                         = splitToInts(layersArg.getValue(),',');
	    o.activationFunction              = activationFunctionArg.getValue();
	    o.samplesFile                       = fileXArg.getValue();
	    o.resultsFile                         = fileYArg.getValue();
        o.numberOfTrainingSamples = numTrainingSamplesArg.getValue();
		o.maxIterations                   = maxIterArg.getValue(); 
        o.alpha                        = alphaArg.getValue(); 
    return o;
}

std::vector<int> &splitToInts(const std::string &s, char delim, 
    std::vector<int> &elems) 
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) 
        elems.push_back(atoi(item.c_str()));
    
    return elems;
}

std::vector<int> splitToInts(const std::string &s, char delim) 
{
    std::vector<int> elems;
    splitToInts(s, delim, elems);
    return elems;
}

void readCsvIntoMatrix(const std::string fileName, float* M, const int rows, 
                        const int columns, bool addBiasColumn)
{
    std::ifstream ifs (fileName.c_str());
	char dummy;
	float x;
	int biasOffset = addBiasColumn ? 1 : 0;
	for (int i = 0; i < rows; ++i){
	    if (addBiasColumn) M[i * (columns + biasOffset)] = 1.0;
		for (int j = 0; j < columns; ++j){
			ifs >> x;
			M[i * (columns + biasOffset) + j + biasOffset] = x; 
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

void printMatrixFromDevice(float *M, int rows, int columns)
{
    float *tmp  = (float *) malloc (sizeof(float) * rows * columns);
	cudaMemcpy(tmp, M, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(tmp, rows, columns);
	free(tmp);
}

void printMatrix(float *M, int rows, int columns)
{
    printf("M:\n");
    for (int i = 0; i < rows; i++){
	    for (int j = 0; j < columns; j++)
	        printf("%f, ", M[i * columns + j]);
	    printf("\n");
	} 
    printf ("##################################################################\
##################################################################\n ");
}

//=============================================================================
// KERNEL WRAPPER FUNCTIONS
//=============================================================================
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

template<typename UnaryFunction>
void CalculateActivation(float* A, float* B, float* C, float* aC, int ARows, int ACols,  int BRows, int BCols, bool addBias, UnaryFunction activationFunction )
{
	dim3 dimBlock(TILE_DIM, TILE_DIM);
	dim3 dimGrid((BCols + dimBlock.x - 1)/ dimBlock.x, (ARows + dimBlock.y - 1)/ dimBlock.y);
                     
    MatMulKernel<<<dimGrid, dimBlock>>>(A, B, C, aC, ARows, ACols, BRows, BCols, ARows, BCols, addBias, activationFunction);
    cudaThreadSynchronize();
}

void MatMul(float* A, float* B, float* C, int ARows, int ACols,  int BRows, int BCols)
{
    dummyFunc dummyf;
	dim3 dimBlock(TILE_DIM, TILE_DIM);
	dim3 dimGrid((BCols + dimBlock.x - 1)/ dimBlock.x, (ARows + dimBlock.y - 1)/ dimBlock.y);
                     
    MatMulKernel<<<dimGrid, dimBlock>>>(A, B, C, NULL, ARows, ACols, BRows, BCols, ARows, BCols, false, dummyf);
    cudaThreadSynchronize();
}
									   
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
	ZipMapReduceKernel<<<dimGrid, dimBlock>>>(d_X, d_Y, d_R, size, mapFunction, 
	                                          neutralElement, reduceFunction);
    cudaThreadSynchronize();
    cudaMemcpy(R, d_R, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);

    //Reduce remaining values in host
    for (int i = 0; i < dimGrid.x; i++)
        r = reduceFunction(r, R[i]);
    cudaFree(d_R);
    free(R);
    return r;
}

template<typename MapFunction>
void ZipMap(float* d_X, float* d_Y, float* d_R, int size, MapFunction mapFunction)
{
    dim3 dimBlock(TILE_DIM);
	dim3 dimGrid((size + dimBlock.x - 1)/ dimBlock.x);
	ZipMapKernel<<<dimGrid, dimBlock>>>(d_X, d_Y, d_R, size, mapFunction);
}

void Transpose(float* d_A, float* d_B, int rows, int cols)
{
    dim3 dimBlock(TILE_DIM, TILE_DIM,1);
    dim3 dimGrid(( cols + TILE_DIM -1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM,1);
	TransposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B, rows, cols);
    cudaThreadSynchronize();
}

//==============================================================================
// KERNELS
//==============================================================================
template<typename UnaryFunction>
__global__ void MatMulKernel(float* A, float* B, float* C, float* aC, int ARows, int ACols, 
                       int BRows, int BCols, int CRows, int CCols, bool addBias, 
                       UnaryFunction activationFunction ) 
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
            Bs[threadIdx.y][threadIdx.x] = 
                B[(k * TILE_DIM + threadIdx.y) * BCols + Col]; 
         else      
            Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n) 
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }
    if (addBias)
	{
		/*__shared__ float BiasRow[TILE_DIM];
	    
		if (threadIdx.y == 0){
		  if (Col < BCols){
			BiasRow[threadIdx.x] = B[Col];
		  }else{
		  	BiasRow[threadIdx.x] = 0.0;
			}
			}
	    __syncthreads();*/
		
		//CValue += BiasRow[threadIdx.x];
		
		//__syncthreads();
		
	}
	
    if (Row < CRows && Col < CCols) {
        if (aC != NULL)
            aC[Row *  (CCols + biasOffset) + Col + biasOffset] = activationFunction(CValue);
        C[Row * CCols + Col] = CValue;
    }
    
    if (addBias)
    {
        if (Row < CRows && (Col == 0))
        {
            aC[Row * (CCols + biasOffset)] = 1.0;
            //C[Row * (CCols + biasOffset) + Col] = 1.0;
        }
    }
    
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

template<typename MapFunction>
__global__ void ZipMapKernel(float* X, float* Y, float* R, int size, 
                                    MapFunction mapFunction)
{
    unsigned int i = blockIdx.x * TILE_DIM + threadIdx.x;

    //Zip and Map: sR <- Map(Zip(sX,sY))
    if (i < size) 
        R[i] = mapFunction(X[i],Y[i]);
    __syncthreads();
}

__global__ void TransposeKernel(float *d_A, float *d_At, int rows, int cols)
{
    __shared__ float block[TILE_DIM][TILE_DIM+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    if((xIndex < cols) && (yIndex < rows))
    {
            unsigned int index_in = yIndex * cols + xIndex;
            block[threadIdx.y][threadIdx.x] = d_A[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    if((xIndex < rows) && (yIndex < cols))
    {
            unsigned int index_out = yIndex * rows + xIndex;
            d_At[index_out] = block[threadIdx.x][threadIdx.y];
    }  
}
