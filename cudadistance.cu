/*
#ifndef __CUDACC__  
#define __CUDACC__
#endif


#include <iostream>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iterator>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SUBSEQLENGTH 10
#define NUMTHREADS 128
#define TSLENGTH 24125                      // Lunghezza txt
//#define TSLENGTH 2299                     // Lunghezza csv
#define NUMBLOCKS (TSLENGTH + NUMTHREADS - 1) / NUMTHREADS              // Calcolo il numero ottimale di blocchi


__constant__ float primo_vettore_confronto[SUBSEQLENGTH];              // Si crea il vettore della memoria constant e lo si riempe con una parte
                                                                       // del vettore timeSeries lungo SUBSEQLENGTH partendo dall'indice i
using namespace std;


__inline__ __device__ void warpReduceMin(float& val, int& idx) {

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        int tmpVal = __shfl_down(val, offset);
        int tmpIdx = __shfl_down(idx, offset);
        if (tmpVal < val) {
            val = tmpVal;
            idx = tmpIdx;
        }
    }
}

__inline__ __device__  void blockReduceMin(float& val, int& idx) {

    static __shared__ int values[32], indices[32];      // Shared mem for 32 partial mins
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceMin(val, idx);                            // Each warp performs partial reduction

    if (lane == 0) {
        values[wid] = val;                              // Write reduced value to shared memory
        indices[wid] = idx;                             // Write reduced value to shared memory
    }

    __syncthreads();                                    // Wait for all partial reductions

    if (threadIdx.x < blockDim.x / warpSize) {          // Read from shared memory only if that warp existed
        val = values[lane];
        idx = indices[lane];
    }
    else {
        val = INT_MAX;
        idx = 0;
    }

    if (wid == 0) {
        warpReduceMin(val, idx);                        //Final reduce within first warp
    }
}


__global__ void finalReduction(int indexFirstSubsequence, float* dev_blocksDistances, int* dev_blocksIndexes, float* dev_finalDistances, int* dev_locations) {

    float val = dev_blocksDistances[threadIdx.x];
    int index = dev_blocksIndexes[threadIdx.x];

    blockReduceMin(val, index);

    dev_finalDistances[indexFirstSubsequence] = val;
    dev_locations[indexFirstSubsequence] = index;
}


__global__ void sequencesDistance(int indexFirstSubsequence, float* dev_timeSeries, float* dev_blocksDistances, int* dev_blocksIndexes) {

    extern __shared__ float cache[];                                               // La sua lunghezza è NUMTHREADS + SUBSEQLENGTH - 1                                      
    float distanza_quadratica;
    //float sum;
    float diff;
    float power;
    int first_arr_index = indexFirstSubsequence;
    int posizione = indexFirstSubsequence;
    int second_arr_index = blockIdx.x * blockDim.x + threadIdx.x;

    int indice_cache = threadIdx.x;

    while (indice_cache < (NUMTHREADS + SUBSEQLENGTH - 1)) {                      // Ogni thread carica nella shared uno o più elementi di timeSeries
        cache[threadIdx.x] = dev_timeSeries[second_arr_index];
        indice_cache = indice_cache + NUMTHREADS;
    }

    if (abs(first_arr_index - second_arr_index) >= SUBSEQLENGTH) {
        for (int i = 0; i < SUBSEQLENGTH; i++) {
            diff = primo_vettore_confronto[i] - cache[threadIdx.x + i];
            power = diff * diff;
            distanza_quadratica = distanza_quadratica + power;
        }
    }
    
    blockReduceMin(distanza_quadratica, posizione);

    if (threadIdx.x == 0) {
        dev_blocksDistances[blockIdx.x] = distanza_quadratica;
        dev_blocksIndexes[blockIdx.x] = posizione;
    }
}


void compareSubsequences(float* timeSeriesHost, float* dev_blocksDistances, int* dev_blocksIndexes, float* dev_timeSeries, float* dev_finalDistances, int* dev_locations) {

    float* support;
    
    support = (float*)malloc(SUBSEQLENGTH * sizeof(float));                             // Vettore di supporto per copiare parte di timeSeries (SUBSEQLENGTH) nella memoria constant

    for (int i = 0; i <= TSLENGTH - SUBSEQLENGTH * 2; i++) {                            // Outer loop
        memcpy(support, &timeSeriesHost[i], sizeof(float) * SUBSEQLENGTH);              // Si copiano in support gli elementi di timeSeriesHost partendo da i
        cudaMemcpyToSymbol(primo_vettore_confronto, support, 10 * sizeof(float), 0, cudaMemcpyHostToDevice);

        sequencesDistance<<<NUMBLOCKS, NUMTHREADS, (NUMTHREADS + SUBSEQLENGTH - 1) * sizeof(float)>>>(i, dev_timeSeries, dev_blocksDistances, dev_blocksIndexes);

        finalReduction<<<1, NUMBLOCKS>>>(i, dev_blocksDistances, dev_blocksIndexes, dev_finalDistances, dev_locations);
    }
    free(support);
    cudaFree(primo_vettore_confronto);
}


//-----------------Lettura e scrittura file INIZIO-----------------------
void readFile(float* timeSeries, string fileNameTxt) {

    const char* c = fileNameTxt.c_str();
    float num = 0;
    int i = 0;
    ifstream readFile;
    readFile.open(c);

    if (!readFile.is_open()) {                                                       // Check to see that the file was opened correctly
        cerr << "There was a problem opening the input file!\n";
        exit(1);                                                                     // Exit or do additional error checking
    }

    while (readFile >> num) {                                                        // Keep storing values from the text file so long as data exists
        timeSeries[i] = float(num);
        i++;
    }

    readFile.close();
}


void scriviFile(float* distances, int* locations, string fileName) {

    FILE* fp;
    char nomeFile;
    //   nomeFile = "nnd.dat";

    fp = fopen("nnd.dat", "w+");
    for (int i = 0; i < TSLENGTH - SUBSEQLENGTH + 1; i++) {
        fprintf(fp, "%lf \n", distances[i]);
    }
    fclose(fp);

    //    nomeFile = "ngh"+fileName  +".dat";
    fp = fopen("loc.dat", "w+");
    for (int i = 0; i < TSLENGTH - SUBSEQLENGTH + 1; i++) {
        fprintf(fp, "%d \n ", locations[i] + 1);
    }
    fclose(fp);
}
//-----------------Lettura e scrittura file FINE--------------------------


int main() {

    string fileNameTxt = "nprs44.txt";
    string fileNameCsv = "ecg0606_1.csv";

    //----------------Blocco TXT start-------------------------------------
        float* timeSeriesTxt;
        float* distancesTxt;
        int* locationsTxt;

        timeSeriesTxt = (float*) malloc(TSLENGTH * sizeof(float));
        distancesTxt = (float*) malloc(TSLENGTH * sizeof(float));
        locationsTxt = (int*) malloc(TSLENGTH * sizeof(int));
        
        fill_n(distancesTxt, TSLENGTH, 9999999999);

        cout << "File name: " << fileNameTxt << endl;
        cout << "File length: " << TSLENGTH << endl;
        cout << "Subsequence length: " << SUBSEQLENGTH << endl;

        readFile(timeSeriesTxt, fileNameTxt);

        float* dev_blocksDistancesTxt;               // Vettore sulla memoria global dove ogni blocco salva il risultato della distanza che calcola
        int* dev_blocksIndexesTxt;
        float* dev_timeSeriesTxt;
        float* dev_finalDistancesTxt;
        int* dev_locationsTxt;

        cudaMalloc((void**)&dev_blocksDistancesTxt, NUMBLOCKS * sizeof(float));
        cudaMalloc((void**)&dev_blocksIndexesTxt, NUMBLOCKS * sizeof(float));
        cudaMalloc((void**)&dev_timeSeriesTxt, TSLENGTH * sizeof(float));
        cudaMalloc((void**)&dev_finalDistancesTxt, TSLENGTH * sizeof(float));
        cudaMalloc((void**)&dev_locationsTxt, TSLENGTH * sizeof(int));
        

        cudaMemcpy(dev_timeSeriesTxt, timeSeriesTxt, TSLENGTH * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_finalDistancesTxt, distancesTxt, TSLENGTH * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_locationsTxt, locationsTxt, TSLENGTH * sizeof(int), cudaMemcpyHostToDevice);

        compareSubsequences(timeSeriesTxt, dev_blocksDistancesTxt, dev_blocksIndexesTxt,dev_timeSeriesTxt, dev_finalDistancesTxt, dev_locationsTxt);     // Funzione che lancia il kernel

        cudaMemcpy(locationsTxt, dev_locationsTxt, TSLENGTH * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(distancesTxt, dev_finalDistancesTxt, TSLENGTH * sizeof(float), cudaMemcpyDeviceToHost);

        scriviFile(distancesTxt, locationsTxt, fileNameTxt);

        free(timeSeriesTxt);
        free(distancesTxt);
        free(locationsTxt);

        cudaFree(dev_blocksDistancesTxt);
        cudaFree(dev_blocksIndexesTxt);
        cudaFree(dev_timeSeriesTxt);
        cudaFree(dev_finalDistancesTxt);
        cudaFree(dev_locationsTxt);
    //----------------Blocco TXT end-------------------------------------
    


    /*
    //----------------Blocco CSV start-------------------------------------
    const int TSLENGTH = 2299;
    
    float* timeSeriesCsv;
    float* distancesCsv;
    int* locationsCsv;

    timeSeriesCsv = (float*) malloc(TSLENGTH * sizeof(float));
    distancesCsv = (float*) malloc(TSLENGTH * sizeof(float));
    locationsCsv = (int*) malloc(TSLENGTH * sizeof(int));

    fill_n(distancesCsv, TSLENGTH, 9999999999);

    cout << "File name: " << fileNameCsv << endl;
    cout << "File length: " << TSLENGTH << endl;
    cout << "Subsequence length: " << SUBSEQLENGTH << endl;

    readFile(timeSeriesCsv, fileNameCsv);

    float* dev_blocksDistancesCsv;               // Vettore sulla memoria global dove ogni thread salva il risultato della distanza che calcola
    float* dev_timeSeriesCsv, 
    float* dev_finalDistancesCsv;
    int* dev_locationsCsv;

    cudaMalloc((void**)&dev_blocksDistancesCsv, TSLENGTH * sizeof(float));
    cudaMalloc((void**)&dev_timeSeriesCsv, TSLENGTH * sizeof(float));
    cudaMalloc((void**)&dev_finalDistancesCsv, TSLENGTH * sizeof(float));
    cudaMalloc((void**)&dev_locationsCsv, TSLENGTH * sizeof(int));

    cudaMemcpy(dev_timeSeriesCsv, timeSeriesCsv, TSLENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_finalDistancesCsv, distancesCsv, TSLENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_locationsCsv, locationsCsv, TSLENGTH * sizeof(int), cudaMemcpyHostToDevice);

    compareSubsequences(timeSeriesCsv, TSLENGTH, dev_blocksDistancesCsv, dev_timeSeriesCsv, dev_finalDistancesCsv, dev_locationsCsv);

    cudaMemcpy(locationsCsv, dev_locationsCsv, TSLENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesCsv, dev_finalDistancesCsv, TSLENGTH * sizeof(float), cudaMemcpyDeviceToHost);

    scriviFile(distancesCsv, locationsCsv, fileNameCsv, TSLENGTH);

    free(timeSeriesCsv);
    free(distancesCsv);
    free(locationsCsv);

    cudaFree(dev_timeSeriesCsv);
    cudaFree(dev_finalDistancesCsv);
    cudaFree(dev_locationsCsv);
    cudaFree(dev_blocksDistancesCsv);
    //----------------Blocco CSV end--------------------------------------
    

    return 0;
}
*/