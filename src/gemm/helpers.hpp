#ifndef RNG_H
#define RNG_H
#include <random>

template <class Vector, class T>
void random_fill(Vector &v)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = T(rand() / float(RAND_MAX));
    }
}

#define CUBLAS_CHECK(call)                                        \
    do                                                            \
    {                                                             \
        cublasStatus_t status = call;                             \
        if (status != CUBLAS_STATUS_SUCCESS)                      \
        {                                                         \
            std::cerr << "cuBLAS Error: " << status << std::endl; \
            return EXIT_FAILURE;                                  \
        }                                                         \
    } while (0)

#endif
