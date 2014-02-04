#include <iostream>
#include <fstream>
#include <tclap/CmdLine.h> //-I wherever TCLAP is
#include <cuda.h>
#include <curand.h>
#include <math.h>

#define TILE_DIM 32
//==============================================================================
// SIGNATURES
//==============================================================================
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


template<typename UnaryFunction>
__global__ void MatMul(float* A, float* B, float* C, float* aC, int ARows, int ACols, 
                       int BRows, int BCols, int CRows, int CCols, bool addBias, 
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
void ZipMap(float* d_X, float* d_Y, float* d_R, int size, MapFunction mapFunction); 

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
};
struct Options ParseCommandLine(int argc, char *argv[]);
void readCsvIntoMatrix(const std::string fileName, float* M, const int rows, 
                        const int columns);
void readResultsIntoMatrix( const std::string fileName, float* M, const int rows, 
                            const int columns);
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A);
void printMatrix(float *M, int rows, int columns);

int main(int argc, char *argv[])
{
    int i,l;
	int lastDelta;
    float *X, *Y, *d_Y, **Theta, **Theta_trans, **d_Theta, **d_Theta_trans, 
          **z, **d_z, **a, **d_a, **a_trans, **d_a_trans, **delta, **d_delta, 
          **Delta, **d_Delta, J; 
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
	//==========================================================================
	// Allocate memory in both host and device
	//==========================================================================
	X = (float *) malloc (sizeof(float) * o.numberOfTrainingSamples * 
	                      o.layerSizes[0]);
	Y = (float *) malloc (sizeof(float) * o.numberOfTrainingSamples * 
	                      o.layerSizes.back());
	cudaMalloc((void **) &d_Y, o.numberOfTrainingSamples * o.layerSizes.back() * 
               sizeof(float));
               
	Theta         = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*)); 
	Theta_trans   = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_Theta       = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_Theta_trans = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	delta         = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_delta       = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	Delta         = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	d_Delta       = (float **) malloc ((o.numberOfLayers - 1) * sizeof(float*));
	z             = (float **) malloc (o.numberOfLayers * sizeof(float*));
	d_z           = (float **) malloc (o.numberOfLayers * sizeof(float*));
	a             = (float **) malloc (o.numberOfLayers * sizeof(float*));
	d_a           = (float **) malloc (o.numberOfLayers * sizeof(float*));
    a_trans       = (float **) malloc (o.numberOfLayers * sizeof(float*));
	d_a_trans     = (float **) malloc (o.numberOfLayers * sizeof(float*));
   
	for (i = 0; i < o.numberOfLayers - 1; i++)
	{
	    Theta[i] = (float *) malloc (sizeof(float) * (o.layerSizes[i] + 1) * 
	                                 o.layerSizes[i+1]); //+1 is the bias row
	    Theta_trans[i] = (float *) malloc (sizeof(float) * (o.layerSizes[i] + 1) * 
	                                 o.layerSizes[i+1]); //+1 is the bias row
	           
	    if (Theta[i] == NULL) printf ("MALLOC ERROR\n");                
        err = cudaMalloc((void **)&(d_Theta[i]), (o.layerSizes[i] + 1) * 
                o.layerSizes[i+1] * sizeof(float)); //+1 is the bias row
        if (err > 0) printf("error code: %d\n",err);
	    
	    //No theta_trans in host by the moment               
	    err = cudaMalloc((void **)&(d_Theta_trans[i]), 
	                     o.numberOfTrainingSamples * 
	                     o.layerSizes[i+1] * sizeof(float));
						 
	    Delta[i] = (float *) malloc (sizeof(float) * (o.layerSizes[i] + 1) * 
	                            (o.layerSizes[i + 1] + 1));            
	    err = cudaMalloc((void **)&(d_Delta[i]), (o.layerSizes[i] + 1) * 
                         (o.layerSizes[i + 1] + 1) * sizeof(float));                                 
	                          
    }
	 //Check the indexes of Delta[1]
	 
	lastDelta = o.numberOfLayers - 2;
	delta[lastDelta] = (float *) malloc (sizeof(float) * o.numberOfTrainingSamples * 
	                                 o.layerSizes.back());
    err = cudaMalloc((void **)&(d_delta[lastDelta]), o.numberOfTrainingSamples * 
	                     o.layerSizes.back() * sizeof(float));
						 
	for (i = 0; i < lastDelta; i++)
	{
		delta[i] = (float *) malloc (sizeof(float) * o.numberOfTrainingSamples * 
										 (o.layerSizes[i+1] + 1));
		err = cudaMalloc((void **)&(d_delta[i]), o.numberOfTrainingSamples * 
							 (o.layerSizes[i+1] + 1) * sizeof(float)); 
	}
	
    for (l = 0; l < o.numberOfLayers; l++) 
    {
        //linear comb for each neuron
        err = cudaMalloc((void **)&(d_z[l]), (o.layerSizes[l] + 1) * 
                o.numberOfTrainingSamples * sizeof(float)); 
        z[l] = (float *) malloc(sizeof(float) * (o.layerSizes[l] + 1) * 
                o.numberOfTrainingSamples); 
        if (err > 0) printf("error code: %d\n",err);
        
        //an activation for each training sample per each neuron at layer l
        err = cudaMalloc((void **)&(d_a[l]), o.layerSizes[l] * 
                o.numberOfTrainingSamples * sizeof(float)); 
        a[l] = (float *) malloc(sizeof(float) * o.layerSizes[l] * 
                o.numberOfTrainingSamples); 
        if (err > 0) printf("error code: %d\n",err);
        
        //an activation for each training sample per each neuron at layer l
        err = cudaMalloc((void **)&(d_a_trans[l]), o.layerSizes[l] * 
                o.numberOfTrainingSamples * sizeof(float)); 
        a_trans[l] = (float *) malloc(sizeof(float) * o.layerSizes[l] * 
                o.numberOfTrainingSamples); 
        if (err > 0) printf("error code: %d\n",err);
        
    }
    
	//==========================================================================
	// INITIALIZE VALUES
	//==========================================================================
	readCsvIntoMatrix(o.samplesFile, X, o.numberOfTrainingSamples, 
                        o.layerSizes[0]);
	readResultsIntoMatrix(o.resultsFile, Y, o.numberOfTrainingSamples, 
                            o.layerSizes.back());
    for (i = 0; i < o.numberOfLayers - 1; i++)
        GPU_fill_rand(d_Theta[i], o.layerSizes[i] + 1, o.layerSizes[i+1]);
	
	/* Force values
	readCsvIntoMatrix("data_theta0.csv", Theta[0], o.layerSizes[0] + 1, 
                        o.layerSizes[1]);
	readCsvIntoMatrix("data_theta1.csv", Theta[1], o.layerSizes[1] + 1, 
                        o.layerSizes[2]);
    cudaMemcpy(d_Theta[0], Theta[0], (o.layerSizes[0] + 1) * o.layerSizes[1] * 
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Theta[1], Theta[1], (o.layerSizes[1] + 1) * o.layerSizes[2] * 
               sizeof(float), cudaMemcpyHostToDevice); */
                        
	//==========================================================================
	// COMPUTE
	//==========================================================================
	printf ("Computing ----------------------------------------------------\n");
	//Feed the X to the first activation function: d_a[0]
	cudaMemcpy(d_a[0], X, o.numberOfTrainingSamples * o.layerSizes[0] * 
	                   sizeof(float), cudaMemcpyHostToDevice);
	//Feed the Y
	cudaMemcpy(d_Y, Y, o.numberOfTrainingSamples * o.layerSizes.back() * 
	                   sizeof(float), cudaMemcpyHostToDevice);
	         
	//dim3 dimBlock(TILE_DIM, TILE_DIM);
	// d_a[1] = d_a[0] x d_Theta[0]
	//multiplication scope, so I can reuse (redeclare) dimGrid ;)
	for(i = 0; i < o.numberOfLayers - 1; i++)
	{ 
		CalculateActivation(d_a[i], d_Theta[i], d_z[i+1], d_a[i+1],
                                     o.numberOfTrainingSamples,
                                     o.layerSizes[i],
                                     o.layerSizes[i],
                                     o.layerSizes[i+1],
                                     sigmoidf);
        /*dim3 dimGrid((o.layerSizes[i+1] + dimBlock.x - 1)/ dimBlock.x, 
                     (o.numberOfTrainingSamples + dimBlock.y - 1)/ dimBlock.y);
                     
        MatMul<<<dimGrid, dimBlock>>>(d_a[i], d_Theta[i], d_z[i+1], d_a[i+1],
                                        o.numberOfTrainingSamples,
                                        o.layerSizes[i],
                                        o.layerSizes[i],
                                        o.layerSizes[i+1],
                                        o.numberOfTrainingSamples,
                                        o.layerSizes[i+1],
                                        true, sigmoidf);
        cudaThreadSynchronize();*/
    }
    //cudaMemcpy(z[1], d_z[1], o.numberOfTrainingSamples * (o.layerSizes[1]+1) * sizeof(float), cudaMemcpyDeviceToHost);
    //printMatrix(z[1], o.numberOfTrainingSamples, o.layerSizes[1]+1);
    /*{ 
    dim3 dimGrid((o.layerSizes[1] + dimBlock.x - 1)/ dimBlock.x, 
                 (o.numberOfTrainingSamples + dimBlock.y - 1)/ dimBlock.y);
                 
    MatMul<<<dimGrid, dimBlock>>>(d_a[0], d_Theta[0], d_a[1],
                                    o.numberOfTrainingSamples,
                                    o.layerSizes[0],
                                    o.layerSizes[0],
                                    o.layerSizes[1],
                                    o.numberOfTrainingSamples,
                                    o.layerSizes[1],
                                    true, sigmoidf);
    cudaThreadSynchronize();
    }
    {
    dim3 dimGrid((o.layerSizes[2] + dimBlock.x - 1) / dimBlock.x, 
                 (o.numberOfTrainingSamples + dimBlock.y - 1)/ dimBlock.y);
                 
    MatMul<<<dimGrid, dimBlock>>>(d_a[1], d_Theta[1], d_a[2],
                                    o.numberOfTrainingSamples,
                                    o.layerSizes[1],
                                    o.layerSizes[1],
                                    o.layerSizes[2],
                                    o.numberOfTrainingSamples,
                                    o.layerSizes[2],
                                    true, sigmoidf);
    cudaThreadSynchronize();
    }*/
    
	
	
    //cudaMemcpy(Theta[0], d_Theta[0], (o.layerSizes[0] + 1) * o.layerSizes[1] * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Theta[1], d_Theta[1], (o.layerSizes[1] + 1) * o.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(a[0], d_a[0], o.numberOfTrainingSamples * o.layerSizes[0] * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(a[1], d_a[1], o.numberOfTrainingSamples * o.layerSizes[1] * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(a[2], d_a[2], o.numberOfTrainingSamples * o.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
    
	//printMatrix(a[0], o.numberOfTrainingSamples, o.layerSizes[0]);
	//printMatrix(Theta[0], o.layerSizes[0]+1, o.layerSizes[1]);
	//printMatrix(a[1], o.numberOfTrainingSamples, o.layerSizes[1]);
	//printMatrix(Theta[1], o.layerSizes[1]+1, o.layerSizes[2]);
	//printMatrix(a[2], o.numberOfTrainingSamples, o.layerSizes[2]);
	//printMatrix(Y,o.numberOfTrainingSamples,o.layerSizes.back());
	
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
	 */
	int lastDelta = o.numberOfLayers - 2;
	
	//Calculate the error in the output: delta[lastDelta] <- d_a[last] - Y
	ZipMap(d_a[o.numberOfLayers - 1], d_Y, d_delta[lastDelta], 
	           o.numberOfTrainingSamples * o.layerSizes.back(), subf);
	       
	//cudaMemcpy(delta[lastDelta], d_delta[lastDelta], o.numberOfTrainingSamples * 
	//           o.layerSizes.back() * sizeof(float), cudaMemcpyDeviceToHost);
	//printMatrix(delta[lastDelta], 100, o.layerSizes.back());
	
	//cudaMemcpy(a[2], d_a[2], o.numberOfTrainingSamples * o.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
	//printMatrix(a[2], 100, o.layerSizes.back());
	
    //(d_delta[lastDelta] x d_Theta_trans[1]) .* g'(z)
    
    Transpose(d_Theta[1], d_Theta_trans[1], o.layerSizes[1] + 1, o.layerSizes[2]);
    
    //cudaMemcpy(Theta_trans[1], d_Theta_trans[1],  (o.layerSizes[1] + 1) * 
    //           o.layerSizes[2] * sizeof(float), cudaMemcpyDeviceToHost);
	//printMatrix(Theta_trans[1], o.layerSizes[2], o.layerSizes[1]+1);
	
	MatMul (d_delta[1], d_Theta_trans[1], d_delta[0],
                o.numberOfTrainingSamples,
                o.layerSizes[2],
                o.layerSizes[2],
                o.layerSizes[1] + 1);
    /*{
    dim3 dimGrid((o.layerSizes[1] + 1 + dimBlock.x - 1)/ dimBlock.x, 
                  (o.numberOfTrainingSamples + dimBlock.y - 1)/ dimBlock.y);
                
    MatMul<<<dimGrid, dimBlock>>>(d_delta[1], d_Theta_trans[1], 
                                  d_delta[0], NULL,
                                  o.numberOfTrainingSamples,
                                  o.layerSizes[2],
                                  o.layerSizes[2],
                                  o.layerSizes[1] + 1,
                                  o.numberOfTrainingSamples,
                                  o.layerSizes[1] + 1,
                                  false, sigmoidf); //dummy functor
    cudaThreadSynchronize();
    }*/
    	
	/*cudaMemcpy(delta[0], d_delta[0], o.numberOfTrainingSamples * 
	          (o.layerSizes[1]+1) * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(delta[0], o.numberOfTrainingSamples, o.layerSizes[1]+1);*/
    
   
    ZipMap(d_z[1], d_z[1], d_z[1], o.numberOfTrainingSamples * (o.layerSizes[1]+1), gradientf);
    ZipMap(d_delta[0], d_z[1], d_delta[0], 
	       o.numberOfTrainingSamples *  (o.layerSizes[1] + 1), prodf);
	
    /*cudaMemcpy(z[1], d_z[1], o.numberOfTrainingSamples * (o.layerSizes[1]+1) * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(z[1], o.numberOfTrainingSamples, o.layerSizes[1]+1);*/
    
   //UP TO THIS POINT EVERYTHING IS FINE
   /*cudaMemcpy(delta[0], d_delta[0], o.numberOfTrainingSamples * 
	           (o.layerSizes[1]+1) * sizeof(float), cudaMemcpyDeviceToHost);
	for(i = 0; i < o.numberOfTrainingSamples; i++) delta[0][i*26] = 0.0;
	
	cudaMemcpy(d_delta[0], delta[0], o.numberOfTrainingSamples * 
	           (o.layerSizes[1]+1) * sizeof(float), cudaMemcpyHostToDevice);*/
	           
/*    cudaMemcpy(delta[0], d_delta[0], o.numberOfTrainingSamples * 
	          (o.layerSizes[1]+1) * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(delta[0], o.numberOfTrainingSamples, o.layerSizes[1]+1);*/

    
    //Transpose(d_a[0], d_a_trans[0], o.numberOfTrainingSamples, o.layerSizes[0]+1);
    Transpose(d_delta[0],d_delta[0], o.numberOfTrainingSamples, o.layerSizes[1]+1);
    //cudaMemcpy(a_trans[0], d_a_trans[0], o.numberOfTrainingSamples 
    //           * o.layerSizes[0] * sizeof(float), cudaMemcpyDeviceToHost);
    //printMatrix(a_trans[0], o.layerSizes[0], o.numberOfTrainingSamples);
    MatMul (d_delta[0], d_a[0], d_Delta[0],
			    o.layerSizes[1] + 1,
			    o.numberOfTrainingSamples,
			    o.numberOfTrainingSamples,
			    o.layerSizes[0] + 1);
	/*{
    dim3 dimGrid((o.layerSizes[0] + 1 + dimBlock.x - 1)/ dimBlock.x, 
                 (o.layerSizes[1] + 1 + dimBlock.y - 1)/ dimBlock.y);
                
    MatMul<<<dimGrid, dimBlock>>>(d_delta[0], d_a[0], 
                                  d_Delta[0], NULL,
                                  o.layerSizes[1] + 1,
                                  o.numberOfTrainingSamples,
                                  o.numberOfTrainingSamples,
                                  o.layerSizes[0] + 1,
                                  o.layerSizes[1] + 1,
                                  o.layerSizes[0] + 1,
                                  false, sigmoidf); //dummy functor
    cudaThreadSynchronize();
    }*/
    
    //Transpose(d_Delta[0], d_Delta[0], o.layerSizes[0] + 1, o.layerSizes[1] + 1);
    cudaMemcpy(Delta[0], d_Delta[0], (o.layerSizes[0] + 1) * (o.layerSizes[1] + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(Delta[0], o.layerSizes[1] + 1, o.layerSizes[0] + 1);
    
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
	for (i = 0; i < o.numberOfLayers - 1; i++) free(Theta[i]);
	for (i = 0; i < o.numberOfLayers - 1; i++) free(Theta_trans[i]);
	for (i = 0; i < o.numberOfLayers - 1; i++) free(delta[i]);
	for (i = 0; i < o.numberOfLayers - 1; i++) free(Delta[i]);
	for (i = 0; i < o.numberOfLayers; i++) free(a[i]);
	for (i = 0; i < o.numberOfLayers; i++) free(z[i]);
	free(Theta);
	free(Theta_trans);
	free(d_Theta);
	free(d_a);
	free(d_z);
	free(d_delta);
	free(Delta);
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
	    
	cmd.parse( argc, argv );
	    o.numberOfLayers          = numLayersArg.getValue();
	    o.layerSizes              = splitToInts(layersArg.getValue(),',');
	    o.activationFunction      = activationFunctionArg.getValue();
	    o.samplesFile             = fileXArg.getValue();
	    o.resultsFile             = fileYArg.getValue();
        o.numberOfTrainingSamples = numTrainingSamplesArg.getValue();
        
    return o;
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
//=============================================================================
// KERNELS WRAPPER FUNCTIONS
//=============================================================================
template<typename UnaryFunction>
void CalculateActivation(float* A, float* B, float* C, float* aC, int ARows, int ACols,  int BRows, int BCols,
									UnaryFunction activationFunction )
{
	dim3 dimBlock(TILE_DIM, TILE_DIM);
	dim3 dimGrid((BCols + dimBlock.x - 1)/ dimBlock.x, (ARows + dimBlock.y - 1)/ dimBlock.y);
                     
    MatMul<<<dimGrid, dimBlock>>>(A, B, C, aC, ARows, ACols, BRows, BCols, ARows, BCols, true, activationFunction);
    cudaThreadSynchronize();
}

void MatMul(float* A, float* B, float* C, int ARows, int ACols,  int BRows, int BCols)
{
    dummyFunc dummyf;
	dim3 dimBlock(TILE_DIM, TILE_DIM);
	dim3 dimGrid((BCols + dimBlock.x - 1)/ dimBlock.x, (ARows + dimBlock.y - 1)/ dimBlock.y);
                     
    MatMul<<<dimGrid, dimBlock>>>(A, B, C, NULL, ARows, ACols, BRows, BCols, ARows, BCols, false, dummyf);
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
                B[(k * TILE_DIM + threadIdx.y + biasOffset) * BCols + Col]; 
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
	
    if (Row < CRows && Col < CCols) {
        if (aC != NULL)
            aC[Row * CCols + Col] = activationFunction(CValue);
        C[Row * (CCols + biasOffset) + Col + biasOffset] = CValue;
    }
    
    if (addBias)
    {
        if (Row < CRows && (Col == 0))
        {
            C[Row * (CCols + biasOffset) + Col] = 1.0;
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
    //__shared__ float sX[TILE_DIM];
    //__shared__ float sY[TILE_DIM];
    //__shared__ float sR[TILE_DIM];
  
    unsigned int i = blockIdx.x * TILE_DIM + threadIdx.x;
    //unsigned int tid = threadIdx.x;
        
    //Load data from memory to shared memory collectively
    //sX[tid] = X[i];
    //sY[tid] = Y[i];
    //__syncthreads();
    
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
