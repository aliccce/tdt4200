#include <stdio.h>
#include "genann_blas.h"

int main(int argc, char *argv[])
{
    printf("GENANN example 1.\n");
    printf("Train a small ANN to the XOR function using backpropagation.\n");

    /* Input and expected out data for the XOR function. */
    const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double output[4] = {0, 1, 1, 0};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */

    //IT MIGHT BE HARD TO GET ANY SPEEDUP WITH A NETWORK THIS SMALL! Scale It up
    //Example : genann *ann = genann_init(2, 2, 512, 1);
    double start, end, start_run, end_run;

    genann *ann = genann_init(2, 2, 512, 1);

    int trainingIter = 2000;

    start = walltime();

    /* Train on the four labeled data points many times. */
    for (i = 0; i < trainingIter; ++i) {
        genann_train(ann, input[0], output + 0, 3);
        genann_train(ann, input[1], output + 1, 3);
        genann_train(ann, input[2], output + 2, 3);
        genann_train(ann, input[3], output + 3, 3);
    }

    end = walltime();

    printf("Time for %d training iterations was : %f\n", trainingIter, end-start);
    printf("Average for each iteration was : %f\n", (end-start)/trainingIter );

    /* Run the network and see what it predicts. */
    double out0, out1, out2, out3;
    start_run = walltime();

    out0 = *genann_run(ann, input[0]);
    out1 = *genann_run(ann, input[1]);
    out2 = *genann_run(ann, input[2]);
    out3 = *genann_run(ann, input[3]);

    end_run = walltime();

    printf("Time for running genann_run: %f\n", end_run-start_run );

    printf("Output for [%1.f, %1.f] is %1.f.\n", input[0][0], input[0][1], out0 );
    printf("Output for [%1.f, %1.f] is %1.f.\n", input[1][0], input[1][1], out1 );
    printf("Output for [%1.f, %1.f] is %1.f.\n", input[2][0], input[2][1], out2 );
    printf("Output for [%1.f, %1.f] is %1.f.\n", input[3][0], input[3][1], out3 );

    genann_free(ann);
    return 0;
}
