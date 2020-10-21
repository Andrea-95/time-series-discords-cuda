/*
#ifndef __CUDACC__
#define __CUDACC__
#endif
*/

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
#define NUMTHREADS 32
#define TSLENGTH 24125                      // Lunghezza txt
//#define TSLENGTH 2299                     // Lunghezza csv
#define NUMBLOCKS (TSLENGTH + NUMTHREADS - 1) / NUMTHREADS            // Calcolo del numero ottimale di blocchi

__constant__ double primo_vettore_confronto[SUBSEQLENGTH];            // Si crea il vettore della memoria constant e lo si riempe con una parte
                                                                      // del vettore timeSeries lungo SUBSEQLENGTH partendo dall'indice i

using namespace std;


__inline__ __device__ void warpReduceMin(double& val, int& idx) {

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        double tmpVal = __shfl_down_sync(0xFFFFFFFF, val, offset, 32);
        int tmpIdx = __shfl_down_sync(0xFFFFFFFF, idx, offset, 32);

        if (tmpVal == val) {                                          // Se due valori di distanza sono uguai si salva la posizione più piccola
            if (tmpIdx < idx) {
                idx = tmpIdx;
            }
        }

        if (tmpVal < val) {
            if (!(tmpVal == 0)) {                                    // TODO verifica dei valori = 0. Alcuni sono corretti in quanto la distanza tra due sottosequenza è effettivamente 0
                val = tmpVal;
                idx = tmpIdx;
            }
        }
    }
}


__inline__ __device__ void blockReduceMin(double& val, int& idx, int currentThreads, int indexFirstSubsequence) {

    static __shared__ double values[32], indices[32];               // Shared mem for 32 partial mins
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceMin(val, idx);                                        // Each warp performs partial reduction

    if (lane == 0) {
        values[wid] = val;                                          // Write reduced value to shared memory
        indices[wid] = idx;                                         // Write reduced value to shared memory
    }

    __syncthreads();                                                // Wait for all partial reductions

    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize && threadIdx.x <= currentThreads / warpSize) {     // Read from shared memory only if that warp existed
        val = values[lane];
        idx = indices[lane];
    }
    else {
        val = DBL_MAX;
        idx = 0;
    }

    if (wid == 0) {
        warpReduceMin(val, idx);                                    // Final reduce within first warp
    }
}


__global__ void sequencesDistance(int indexFirstSubsequence, double* dev_timeSeries, double* dev_blocksDistances, int* dev_blocksLocations) {

    extern __shared__ double cache[];                                    // La sua lunghezza è NUMTHREADS + SUBSEQLENGTH - 1   
    double sum = 0;
    double distanza = DBL_MAX;                                           // Variabile a cui è assegnata la distanza. I thread non coinvolti nel calcolo hanno questo valore come default
    double diff;
    double power;

    int first_arr_index = indexFirstSubsequence;                         // Indice della prima sottosequenza che verrà confrontata con tutte le altre
    int second_arr_index = blockIdx.x * blockDim.x + threadIdx.x;        // Indice della seconda sottosequenza che si confronta con la prima. L'id globale di ogni thread stabilisce il punto di partenza
    int altro_indice = blockIdx.x * blockDim.x + threadIdx.x;
    int indice_cache = threadIdx.x;


    while (indice_cache < (NUMTHREADS + SUBSEQLENGTH - 1) && altro_indice <= TSLENGTH) {       // Ogni thread carica nella shared uno o più elementi di timeSeries
        cache[indice_cache] = dev_timeSeries[altro_indice];
        indice_cache = indice_cache + NUMTHREADS;
        altro_indice = altro_indice + NUMTHREADS;
    }

    __syncthreads();                                                     // SYNC perché non tutti possono aver già riempito la cache  


    if (abs(second_arr_index - first_arr_index) >= SUBSEQLENGTH && second_arr_index < TSLENGTH - SUBSEQLENGTH + 1) {    // Verifica di self-match e controllo che il secondo indice rientri nei valori consentiti
        for (int i = 0; i < SUBSEQLENGTH; i++) {
            diff = primo_vettore_confronto[i] - cache[threadIdx.x + i];
            power = diff * diff;
            sum = sum + power;
        }
        distanza = sqrt(sum);                                            // Da rimuovere nella versione finale, la radice si calcola alla fine
    }

    blockReduceMin(distanza, second_arr_index, NUMTHREADS, indexFirstSubsequence);

    if (threadIdx.x == 0) {
        dev_blocksDistances[blockIdx.x] = distanza;
        dev_blocksLocations[blockIdx.x] = second_arr_index;
    }
}


__global__ void finalReduction(int indexFirstSubsequence, int previousBlocks, int* dev_blocksLocations, double* dev_blocksDistances, int* dev_finalLocations, double* dev_finalDistances) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < previousBlocks) {
        double val = dev_blocksDistances[tid];
        int idx = dev_blocksLocations[tid];

        blockReduceMin(val, idx, previousBlocks, indexFirstSubsequence);

        if (tid == 0 && gridDim.x != 1) {           // Si utilizzano i vettori dev_blocksDistances e dev_blocksLocations per salvare i risultati delle riduzioni
            dev_blocksDistances[blockIdx.x] = val;  // ad ogni nuova iterazione finché si utilizza più di un blocco 
            dev_blocksLocations[blockIdx.x] = idx;
        }

        if (tid == 0 && gridDim.x == 1) {               // Si scrive sui vettori finali solo quando la riduzione è arrivata ad utilizzare un solo blocco
            dev_finalDistances[indexFirstSubsequence] = val;
            dev_finalLocations[indexFirstSubsequence] = idx;
        }
    }
}


void compareSubsequences(double* dev_blocksDistances, int* dev_blocksLocations, double* dev_timeSeries, double* dev_finalDistances, int* dev_finalLocations) {

    int threads = 1024;

    for (int i = 0; i <= TSLENGTH - SUBSEQLENGTH; i++) {        // Outer loop

        int previousBlocks = NUMBLOCKS;
        int currentBlocks = (NUMBLOCKS + threads - 1) / threads;

        bool continueReduction = true;

        cudaMemcpyToSymbol(primo_vettore_confronto, &dev_timeSeries[i], SUBSEQLENGTH * sizeof(double), 0, cudaMemcpyDeviceToDevice);  // Copia nella constant la sottosequenza all'i-esima posizione 
                                                                                                                                      // da confrontare con tutte le altre

        sequencesDistance << <NUMBLOCKS, NUMTHREADS, (NUMTHREADS + SUBSEQLENGTH - 1) * sizeof(double) >> > (i, dev_timeSeries, dev_blocksDistances, dev_blocksLocations);  // Kernel che esegue il calcolo delle distanze 
                                                                                                                                                                      // ed una prima riduzione

        while (continueReduction) {
            finalReduction << <currentBlocks, threads >> > (i, previousBlocks, dev_blocksLocations, dev_blocksDistances, dev_finalLocations, dev_finalDistances); // Riduce i risultati ottenuti dal kernel precedente

            if (currentBlocks == 1) {
                continueReduction = false;
            }

            previousBlocks = currentBlocks;
            currentBlocks = (currentBlocks + threads - 1) / threads;
        }
    }
    cudaFree(primo_vettore_confronto);
}


//--------------------Lettura e scrittura file start--------------------------
void readFile(double* timeSeries, string fileName) {

    const char* c = fileName.c_str();
    double num = 0;
    int i = 0;
    ifstream readFile;
    readFile.open(c);

    if (!readFile.is_open()) {                                                       // Check to see that the file was opened correctly
        cerr << "There was a problem opening the input file!\n";
        exit(1);                                                                     // Exit or do additional error checking
    }

    while (readFile >> num) {                                                        // Keep storing values from the text file so long as data exists
        timeSeries[i] = double(num);
        i++;
    }

    readFile.close();
}


void scriviFile(double* distances, int* locations, string fileName) {

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

    fp = fopen("total.dat", "w+");
    for (int i = 0; i < TSLENGTH - SUBSEQLENGTH + 1; i++) {
        fprintf(fp, "%lf \t %d \n", distances[i], locations[i] + 1);
    }
    fclose(fp);
}
//--------------------Lettura e scrittura file end-----------------------------


int main() {

    //string fileName = "ecg0606_1.csv";
    string fileName = "nprs44.txt";

    double* timeSeries;
    double* distances;
    int* locations;

    timeSeries = (double*)malloc(TSLENGTH * sizeof(double));
    distances = (double*)malloc(TSLENGTH * sizeof(double));
    locations = (int*)malloc(TSLENGTH * sizeof(int));

    fill_n(distances, TSLENGTH, DBL_MAX);

    cout << "File name: " << fileName << endl;
    cout << "File length: " << TSLENGTH << endl;
    cout << "Subsequence length: " << SUBSEQLENGTH << endl;
    cout << "Number of blocks: " << NUMBLOCKS << endl;
    cout << "Number of threads: " << NUMTHREADS << endl;

    readFile(timeSeries, fileName);

    double* dev_blocksDistances;                 // Vettore sulla memoria global dove ogni blocco salva il risultato della distanza che calcola
    int* dev_blocksLocations;                    // Vettore sulla memoria global dove ogni blocco salva l'indice della propria migliore distanza trovata 
    double* dev_finalDistances;
    int* dev_finalLocations;
    double* dev_timeSeries;

    cudaMalloc((void**)&dev_blocksDistances, NUMBLOCKS * sizeof(double));
    cudaMalloc((void**)&dev_blocksLocations, NUMBLOCKS * sizeof(int));
    cudaMalloc((void**)&dev_timeSeries, TSLENGTH * sizeof(double));
    cudaMalloc((void**)&dev_finalDistances, TSLENGTH * sizeof(double));
    cudaMalloc((void**)&dev_finalLocations, TSLENGTH * sizeof(int));

    cudaMemcpy(dev_timeSeries, timeSeries, TSLENGTH * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_finalDistances, distances, TSLENGTH * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_finalLocations, locations, TSLENGTH * sizeof(int), cudaMemcpyHostToDevice);

    compareSubsequences(dev_blocksDistances, dev_blocksLocations, dev_timeSeries, dev_finalDistances, dev_finalLocations);     // Funzione che lancia il kernel

    cudaMemcpy(locations, dev_finalLocations, TSLENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, dev_finalDistances, TSLENGTH * sizeof(double), cudaMemcpyDeviceToHost);

    scriviFile(distances, locations, fileName);

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
