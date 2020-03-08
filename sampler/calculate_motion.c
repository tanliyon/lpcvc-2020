//
//  calculate_motion.c
//  
//
//  Created by Fischer Bordwell on 2/26/20.
//

#include <stdio.h>

int main(int argc, char** argv) {
    long sum = 0;
    long i = 0;
    while(scanf("%ld", &i) == 1) {
        sum = sum + i;
    }
    printf("%ld\n", sum);
    return 0;
}
