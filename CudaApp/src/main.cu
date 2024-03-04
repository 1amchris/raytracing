#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "helper.h"

int main() {

    add_vectors_and_print_them(1000);

    return 0;
}
