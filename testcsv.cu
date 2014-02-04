#include <iostream>
#include <fstream>

void readCsvIntoMatrix(const std::string fileName, float* M, const int rows, 
                        const int columns)
{
    std::ifstream ifs (fileName.c_str());
	char dummy;
	float x;
	
	for (int i = 0; i < rows; ++i){
		for (int j = 0; j < columns; ++j){
			ifs >> x;
			printf ("%f, ",x);
			M[i * columns + j] = x; 
			if (j < (columns - 1)) //ignore commas
				ifs >> dummy;
		}
		printf("\n");
	}
}


int main(int argc, char *argv[])
{
    float* Theta1 =  (float *) malloc (sizeof(float) * 26 * 10);
    readCsvIntoMatrix("data_theta1.csv", Theta1, 26, 10);
    return 0;                       
}
