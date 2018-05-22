#include <iostream>
#include <chrono>

class Timer
{
public:
    Timer() : _beg(_clock::now()) {}
    void reset() { _beg = _clock::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<_second>
            (_clock::now() - _beg).count(); }

private:
    typedef std::chrono::high_resolution_clock _clock;
    typedef std::chrono::duration<double, std::ratio<1> > _second;
    std::chrono::time_point<_clock> _beg;
};
