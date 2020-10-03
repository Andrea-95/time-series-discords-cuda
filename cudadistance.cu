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

#include <algorithm>

#define SUBSEQLENGTH 10
#define NUMTHREADS 128
#define TSLENGTH 24125                      // Lunghezza txt
//#define TSLENGTH 2299                     // Lunghezza csv
#define NUMBLOCKS (TSLENGTH + NUMTHREADS - 1) / NUMTHREADS             // Calcolo del numero ottimale di blocchi

__constant__ float primo_vettore_confronto[SUBSEQLENGTH];              // Si crea il vettore della memoria constant e lo si riempe con una parte
                                                                       // del vettore timeSeries lungo SUBSEQLENGTH partendo dall'indice i
using namespace std;


__inline__ __device__ void warpReduceMin(float& val, int& idx) {

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float tmpVal = __shfl_down_sync(0xFFFFFFFF, val, offset, 32);
        int tmpIdx = __shfl_down_sync(0xFFFFFFFF, idx, offset, 32);
        if (tmpVal < val) {
            val = tmpVal;
            idx = tmpIdx;
        }
    }
}

__inline__ __device__ void blockReduceMin(float& val, int& idx) {

    static __shared__ float values[32], indices[32];    // Shared mem for 32 partial mins
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
        warpReduceMin(val, idx);                        // Final reduce within first warp
    }
}


__global__ void sequencesDistance(int indexFirstSubsequence, float* dev_timeSeries, float* dev_blocksDistances, int* dev_blocksLocations, int* dev_finalLocations, float* dev_finalDistances) {

    extern __shared__ float cache[];                                                    // La sua lunghezza è NUMTHREADS + SUBSEQLENGTH - 1                                      
    float distanza_quadratica;
    float distanza;
    float diff;
    float power;
    int first_arr_index = indexFirstSubsequence;
    int second_arr_index = blockIdx.x * blockDim.x + threadIdx.x;

    int indice_cache = threadIdx.x;

    while (indice_cache < (NUMTHREADS + SUBSEQLENGTH - 1)) {                            // Ogni thread carica nella shared uno o più elementi di timeSeries
        cache[indice_cache] = dev_timeSeries[second_arr_index];
        indice_cache = indice_cache + NUMTHREADS;
    }

    __syncthreads();                                                                    // SYNC perché non tutti possono aver già riempito la cache  

    if (abs(first_arr_index - second_arr_index) >= SUBSEQLENGTH) {
        for (int i = 0; i < SUBSEQLENGTH; i++) {
            diff = primo_vettore_confronto[i] - cache[threadIdx.x + i];
            power = diff * diff;
            distanza_quadratica = distanza_quadratica + power;
        }
    }
    distanza = sqrt(distanza_quadratica);                                               // Da rimuovere nella versione finale, la radice si calcola alla fine


    dev_finalDistances[blockIdx.x * blockDim.x + threadIdx.x] = distanza;               // Per salvare tutte le distanze calcolate rispetto ad una sottosequenza
    dev_finalLocations[blockIdx.x * blockDim.x + threadIdx.x] = second_arr_index;       // tenere queste due righe

    /*
    blockReduceMin(distanza_quadratica, second_arr_index);                              // Per salvare solo le distanzze migliori di ogni blocco tenere questa riga

    if (threadIdx.x == 0) {                                                             // e anche il blocco dell'if
        dev_blocksDistances[blockIdx.x] = sqrt(distanza_quadratica);
        dev_blocksLocations[blockIdx.x] = second_arr_index;
    }
    
}


__global__ void finalReduction(int indexFirstSubsequence, int* dev_blocksLocations, float* dev_blocksDistances, int* dev_finalLocations, float* dev_finalDistances) {


    float val = dev_blocksDistances[threadIdx.x];
    int index = dev_blocksLocations[threadIdx.x];

    blockReduceMin(val, index);

    if (threadIdx.x == 0) {
        dev_finalDistances[indexFirstSubsequence] = val;
        dev_finalLocations[indexFirstSubsequence] = index;
    }


    /*
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            if (dev_blocksDistances[threadIdx.x] < dev_blocksDistances[threadIdx.x + i]) {
                dev_finalDistances[threadIdx.x] = dev_blocksDistances[threadIdx.x];
            }
            else {
                dev_finalDistances[threadIdx.x] =  dev_blocksDistances[threadIdx.x + i];
            }
        }
        __syncthreads();
        i /= 2;
    }
    
}


void compareSubsequences(float* timeSeriesHost, float* dev_blocksDistances, int* dev_blocksLocations, float* dev_timeSeries, float* dev_finalDistances, int* dev_finalLocations) {

    float* support;
    support = (float*)malloc(SUBSEQLENGTH * sizeof(float));                                 // Vettore di supporto per copiare parte di timeSeries (SUBSEQLENGTH) nella memoria constant

    //for (int i = 0; i <= TSLENGTH - SUBSEQLENGTH * 2; i++) {                              // Outer loop, se commentato si testa il kernel su una sola sottosequenza
    memcpy(support, &timeSeriesHost[1], sizeof(float) * SUBSEQLENGTH);                  // Si copiano in support gli elementi di timeSeriesHost partendo da 0
    cudaMemcpyToSymbol(primo_vettore_confronto, support, SUBSEQLENGTH * sizeof(float), 0, cudaMemcpyHostToDevice);

    sequencesDistance << <NUMBLOCKS, NUMTHREADS, (NUMTHREADS + SUBSEQLENGTH - 1) * sizeof(float) >> > (1, dev_timeSeries, dev_blocksDistances, dev_blocksLocations, dev_finalLocations, dev_finalDistances);

    //finalReduction<<<1, NUMBLOCKS>>>(0, dev_blocksLocations, dev_blocksDistances, dev_finalLocations, dev_finalDistances);
//}
    free(support);
    cudaFree(primo_vettore_confronto);
}


//--------------------Lettura e scrittura file start--------------------------
void readFile(float* timeSeries, string fileName) {

    const char* c = fileName.c_str();
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
//--------------------Lettura e scrittura file end-----------------------------


int main() {

    //string fileName = "ecg0606_1.csv";
    string fileName = "nprs44.txt";

    float* timeSeries;
    float* distances;
    int* locations;

    timeSeries = (float*)malloc(TSLENGTH * sizeof(float));
    distances = (float*)malloc(TSLENGTH * sizeof(float));
    locations = (int*)malloc(TSLENGTH * sizeof(int));

    fill_n(distances, TSLENGTH, 9999999999);

    cout << "File name: " << fileName << endl;
    cout << "File length: " << TSLENGTH << endl;
    cout << "Subsequence length: " << SUBSEQLENGTH << endl;
    cout << "Number of blocks: " << NUMBLOCKS << endl;
    cout << "Number of threads: " << NUMTHREADS << endl;

    readFile(timeSeries, fileName);

    float* dev_blocksDistances;                 // Vettore sulla memoria global dove ogni blocco salva il risultato della distanza che calcola
    int* dev_blocksLocations;                   // Vettore sulla memoria global dove ogni blocco salva l'indice della propria migliore distanza trovata 
    float* dev_finalDistances;
    int* dev_finalLocations;
    float* dev_timeSeries;

    cudaMalloc((void**)&dev_blocksDistances, NUMBLOCKS * sizeof(float));
    cudaMalloc((void**)&dev_blocksLocations, NUMBLOCKS * sizeof(int));
    cudaMalloc((void**)&dev_timeSeries, TSLENGTH * sizeof(float));
    cudaMalloc((void**)&dev_finalDistances, TSLENGTH * sizeof(float));
    cudaMalloc((void**)&dev_finalLocations, TSLENGTH * sizeof(int));

    cudaMemcpy(dev_timeSeries, timeSeries, TSLENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_finalDistances, distances, TSLENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_finalLocations, locations, TSLENGTH * sizeof(int), cudaMemcpyHostToDevice);

    compareSubsequences(timeSeries, dev_blocksDistances, dev_blocksLocations, dev_timeSeries, dev_finalDistances, dev_finalLocations);     // Funzione che lancia il kernel

    cudaMemcpy(locations, dev_finalLocations, TSLENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, dev_finalDistances, TSLENGTH * sizeof(float), cudaMemcpyDeviceToHost);

    float temp;
    for (int i = 0; i < TSLENGTH; i++)
    {
        for (int j = i + 1; j < TSLENGTH; j++)
        {
            if (distances[i] > distances[j])
            {
                temp = distances[i];
                distances[i] = distances[j];
                distances[j] = temp;
            }
        }
    }

    scriviFile(distances, locations, fileName);         // Testare con sottosequenza > n thread in un blocco

    free(timeSeries);
    free(distances);
    free(locations);

    cudaFree(dev_blocksDistances);
    cudaFree(dev_blocksLocations);
    cudaFree(dev_timeSeries);
    cudaFree(dev_finalDistances);
    cudaFree(dev_finalLocations);

    return 0;
}
*/