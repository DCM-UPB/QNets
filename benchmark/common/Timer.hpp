#include <chrono>
#include <iostream>

class Timer
{
public:
    explicit Timer(const double scale) : _beg(_clock::now()), _scale(scale) {}
    void reset() { _beg = _clock::now(); }
    double elapsed() const {
        using second = std::chrono::duration<double, std::ratio<1>>;
        return _scale * std::chrono::duration_cast<second>(_clock::now() - _beg).count();
    }

private:
    using _clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<_clock> _beg;
    const double _scale;
};
