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

#endif
