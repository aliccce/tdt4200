#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define dT 0.2f
#define G 0.6f
#define BLOCK_SIZE 64

// Global variables
int num_planets;
int num_timesteps;

// Host arrays
float2* velocities;
float4* planets;

// Device arrays 
float2* velocities_d;
float4* planets_d;


// Parse command line arguments
void parse_args(int argc, char** argv){
    if(argc != 2){
        printf("Useage: nbody num_timesteps\n");
        exit(-1);
    }
    
    num_timesteps = strtol(argv[1], 0, 10);
}

// Reads planets from planets.txt
void read_planets(){

    FILE* file = fopen("planets.txt", "r");
    if(file == NULL){
        printf("'planets.txt' not found. Exiting\n");
        exit(-1);
    }

    char line[200];
    fgets(line, 200, file);
    sscanf(line, "%d", &num_planets);

    planets = (float4*)malloc(sizeof(float4)*num_planets);
    velocities = (float2*)malloc(sizeof(float2)*num_planets);

    for(int p = 0; p < num_planets; p++){
        fgets(line, 200, file);
        sscanf(line, "%f %f %f %f %f",
                &planets[p].x,
                &planets[p].y,
                &velocities[p].x,
                &velocities[p].y,
                &planets[p].z);
    }

    fclose(file);
}

// Writes planets to file
void write_planets(int timestep){
    char name[20];
    int n = sprintf(name, "planets_out.txt");

    FILE* file = fopen(name, "wr+");

    for(int p = 0; p < num_planets; p++){
        fprintf(file, "%f %f %f %f %f\n",
                planets[p].x,
                planets[p].y,
                velocities[p].x,
                velocities[p].y,
                planets[p].z);
    }

    fclose(file);
}

// TODO 7. Calculate the change in velocity for p, caused by the interaction with q
__device__ float2 calculate_velocity_change_planet(float4 p, float4 q){

}

// TODO 5. Calculate the change in velocity for my_planet, caused by the interactions with a block of planets
__device__ float2 calculate_velocity_change_block(float4 my_planet, float4* shared_planets){

}

// TODO 4. Update the velocities by calculating the planet interactions
__global__ void update_velocities(float4* planets, float2* velocities, int num_planets){

}

// TODO 7. Update the positions of the planets using the new velocities
__global__ void update_positions(float4* planets, float2* velocities, int num_planets){

}


int main(int argc, char** argv){

    parse_args(argc, argv);
    read_planets();

    // TODO 1. Allocate device memory, and transfer data to device 

    // Calculating the number of blocks
    int num_blocks = num_planets/BLOCK_SIZE + ((num_planets%BLOCK_SIZE == 0) ? 0 : 1);

    // Main loop
    for(int t = 0; t < num_timesteps; t++){
        // TODO 2. Call kernels
    }

    // TODO 3. Transfer data back to host

    // Output
    write_planets(num_timesteps);
}
