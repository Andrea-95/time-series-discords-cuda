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

#define SUBSEQLENGTH 10
#define NUMBLOCKS 400
#define NUMTHREADS 128

using namespace std;


__global__ void sequencesDistance(int indexFirstSubsequence, int tsLength, float* dev_blocksDistances, float* dev_timeSeries, float* dev_finalDistances, int* dev_locations) {

    extern __shared__ float cache[];                                                
    //float threadDist = 99999999;
    //int threadLoc;
    float distanza_quadratica;
    float sum;
    float diff;
    float power;
    int first = indexFirstSubsequence;
    int second = blockIdx.x * blockDim.x + threadIdx.x;

    cache[threadIdx.x] = dev_timeSeries[second];                                      // Ogni thread carica parte di timeSeries nella memoria shared

    //while (second < tsLength - subSeqLength + 1) {

    if (abs(first - second) >= SUBSEQLENGTH) {
        for (int i = 0; i < SUBSEQLENGTH; i++) {
            diff = primo_vettore_confronto[i] - cache[second + i];
            power = diff * diff;
            distanza_quadratica = distanza_quadratica + power;
        }
    }

    dev_blocksDistances[threadIdx.x] = distanza_quadratica;
    /*
    distance = sqrt(sum);

    if (distance < threadDist) {
        threadDist = distance;
        threadLoc = second;
    }
            
    if (distance < dev_distancesTxt[first]) {
        //dev_distancesTxt[first] = distance;
        //dev_locationsTxt[first] = second;

        atomicExch(&dev_distancesTxt[first], distance);
        atomicExch(&dev_locationsTxt[first], second);
    }

    if (distance < dev_distancesTxt[second]) {
        dev_distancesTxt[second] = distance;
        dev_locationsTxt[second] = first;

        //atomicExch(&dev_distancesTxt[second], distance);
        //atomicExch(&dev_locationsTxt[second], first);
    }
    }
    second += blockDim.x * gridDim.x;
    sum = 0;    
    }
    */
    __syncthreads();

}


void compareSubsequences(float* timeSeriesHost, int tsLength, float* dev_blocksDistances, float* dev_timeSeries, float* dev_finalDistances, int* dev_locations) {

    __constant__ float primo_vettore_confronto[SUBSEQLENGTH];                           // Si crea il vettore della memoria constant e lo si riempe con una parte
                                                                                        // del vettore timeSeries lungo SUBSEQLENGTH partendo dall'indice i
    float* support;
    support = (float*)malloc(tsLength * sizeof(float));                                 // Vettore di supporto per copiare parte di timeSeries nella memoria constant

    for (int i = 0; i <= tsLength - SUBSEQLENGTH * 2; i++) {                            // Outer loop

        memcpy(support, &timeSeriesHost[i], sizeof(float) * SUBSEQLENGTH);              // Si copiano in support gli elementi di timeSeriesHost partendo da i
        cudaMemcpyToSymbol(primo_vettore_confronto, support, 10 * sizeof(float), 0, cudaMemcpyHostToDevice);
        sequencesDistance<<<NUMBLOCKS, NUMTHREADS, (NUMTHREADS + SUBSEQLENGTH - 1) * sizeof(float)>>>(i, tsLength, dev_blocksDistances, dev_timeSeries, dev_finalDistances, dev_locations);
    }
    free(support);
}


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


void scriviFile(float* distances, int* locations, string fileName, int tsLength) {

    FILE* fp;
    char nomeFile;
    //   nomeFile = "nnd.dat";

    fp = fopen("nnd.dat", "w+");
    for (int i = 0; i < tsLength - SUBSEQLENGTH + 1; i++) {
        fprintf(fp, "%lf \n", distances[i]);
    }
    fclose(fp);

    //    nomeFile = "ngh"+fileName  +".dat";
    fp = fopen("loc.dat", "w+");
    for (int i = 0; i < tsLength - SUBSEQLENGTH + 1; i++) {
        fprintf(fp, "%d \n ", locations[i] + 1);
    }
    fclose(fp);
}


int main() {

    string fileNameTxt = "nprs44.txt";
    string fileNameCsv = "ecg0606_1.csv";

    //----------------Blocco TXT start-------------------------------------
        const int tsLengthTxt = 24125;

        float* timeSeriesTxt;
        float* distancesTxt;
        int* locationsTxt;

        timeSeriesTxt = (float*) malloc(tsLengthTxt*sizeof(float));
        distancesTxt = (float*) malloc(tsLengthTxt * sizeof(float));
        locationsTxt = (int*) malloc(tsLengthTxt * sizeof(int));
        
        fill_n(distancesTxt, tsLengthTxt, 9999999999);

        cout << "File name: " << fileNameTxt << endl;
        cout << "File length: " << tsLengthTxt << endl;
        cout << "Subsequence length: " << SUBSEQLENGTH << endl;

        readFile(timeSeriesTxt, fileNameTxt);

        float* dev_blocksDistancesTxt;               // Vettore sulla memoria global dove ogni thread salva il risultato della distanza che calcola
        float* dev_timeSeriesTxt;
        float* dev_finalDistancesTxt;
        int* dev_locationsTxt;

        cudaMalloc((void**)&dev_blocksDistancesTxt, tsLengthTxt * sizeof(float));
        cudaMalloc((void**)&dev_timeSeriesTxt, tsLengthTxt * sizeof(float));
        cudaMalloc((void**)&dev_finalDistancesTxt, tsLengthTxt * sizeof(float));
        cudaMalloc((void**)&dev_locationsTxt, tsLengthTxt * sizeof(int));
        

        cudaMemcpy(dev_timeSeriesTxt, timeSeriesTxt, tsLengthTxt * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_finalDistancesTxt, distancesTxt, tsLengthTxt * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_locationsTxt, locationsTxt, tsLengthTxt * sizeof(int), cudaMemcpyHostToDevice);

        compareSubsequences(timeSeriesTxt, tsLengthTxt, dev_blocksDistancesTxt, dev_timeSeriesTxt, dev_finalDistancesTxt, dev_locationsTxt);     // Funzione che lancia il kernel

        cudaMemcpy(locationsTxt, dev_locationsTxt, tsLengthTxt * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(distancesTxt, dev_finalDistancesTxt, tsLengthTxt * sizeof(float), cudaMemcpyDeviceToHost);

        scriviFile(distancesTxt, locationsTxt, fileNameTxt, tsLengthTxt);

        free(timeSeriesTxt);
        free(distancesTxt);
        free(locationsTxt);

        cudaFree(dev_timeSeriesTxt);
        cudaFree(dev_finalDistancesTxt);
        cudaFree(dev_locationsTxt);
        cudaFree(dev_blocksDistancesTxt);
    //----------------Blocco TXT end-------------------------------------
    


    /*
    //----------------Blocco CSV start-------------------------------------
    const int tsLengthCsv = 2299;
    
    float* timeSeriesCsv;
    float* distancesCsv;
    int* locationsCsv;

    timeSeriesCsv = (float*) malloc(tsLengthCsv * sizeof(float));
    distancesCsv = (float*) malloc(tsLengthCsv * sizeof(float));
    locationsCsv = (int*) malloc(tsLengthCsv * sizeof(int));

    fill_n(distancesCsv, tsLengthCsv, 9999999999);

    cout << "File name: " << fileNameCsv << endl;
    cout << "File length: " << tsLengthCsv << endl;
    cout << "Subsequence length: " << SUBSEQLENGTH << endl;

    readFile(timeSeriesCsv, fileNameCsv);

    float* dev_blocksDistancesCsv;               // Vettore sulla memoria global dove ogni thread salva il risultato della distanza che calcola
    float* dev_timeSeriesCsv, 
    float* dev_finalDistancesCsv;
    int* dev_locationsCsv;

    cudaMalloc((void**)&dev_blocksDistancesCsv, tsLengthCsv * sizeof(float));
    cudaMalloc((void**)&dev_timeSeriesCsv, tsLengthCsv * sizeof(float));
    cudaMalloc((void**)&dev_finalDistancesCsv, tsLengthCsv * sizeof(float));
    cudaMalloc((void**)&dev_locationsCsv, tsLengthCsv * sizeof(int));

    cudaMemcpy(dev_timeSeriesCsv, timeSeriesCsv, tsLengthCsv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_finalDistancesCsv, distancesCsv, tsLengthCsv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_locationsCsv, locationsCsv, tsLengthCsv * sizeof(int), cudaMemcpyHostToDevice);

    compareSubsequences(timeSeriesCsv, tsLengthCsv, dev_blocksDistancesCsv, dev_timeSeriesCsv, dev_finalDistancesCsv, dev_locationsCsv);

    cudaMemcpy(locationsCsv, dev_locationsCsv, tsLengthCsv * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distancesCsv, dev_finalDistancesCsv, tsLengthCsv * sizeof(float), cudaMemcpyDeviceToHost);

    scriviFile(distancesCsv, locationsCsv, fileNameCsv, tsLengthCsv);

    free(timeSeriesCsv);
    free(distancesCsv);
    free(locationsCsv);

    cudaFree(dev_timeSeriesCsv);
    cudaFree(dev_finalDistancesCsv);
    cudaFree(dev_locationsCsv);
    cudaFree(dev_blocksDistancesCsv);
    //----------------Blocco CSV end--------------------------------------
    */

    return 0;
}
