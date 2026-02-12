#pragma once

// Lightweight compatibility layer that mirrors the ac_std_float API surface
// used by this simulator. Arithmetic delegates to host FP operators.

enum ac_q_mode {
    AC_RND_CONV = 0,
};

template<int W, int E>
class ac_std_float {
public:
    ac_std_float() : v_(0.0) {}
    explicit ac_std_float(double f) : v_(f) {}
    explicit ac_std_float(float f) : v_(f) {}
    explicit ac_std_float(int x) : v_(x) {}

    template<int W2, int E2>
    explicit ac_std_float(const ac_std_float<W2, E2>& f) : v_(f.to_double()) {}

    ac_std_float operator+(const ac_std_float& op2) const { return ac_std_float(v_ + op2.v_); }
    ac_std_float operator-(const ac_std_float& op2) const { return ac_std_float(v_ - op2.v_); }
    ac_std_float operator*(const ac_std_float& op2) const { return ac_std_float(v_ * op2.v_); }
    ac_std_float operator/(const ac_std_float& op2) const { return ac_std_float(v_ / op2.v_); }

    ac_std_float& operator+=(const ac_std_float& op2) {
        v_ += op2.v_;
        return *this;
    }
    ac_std_float& operator*=(const ac_std_float& op2) {
        v_ *= op2.v_;
        return *this;
    }

    template<int WR, int ER, ac_q_mode QR>
    ac_std_float<WR, ER> convert() const {
        (void)QR;
        return ac_std_float<WR, ER>(v_);
    }

    double to_double() const { return v_; }
    float to_float() const { return static_cast<float>(v_); }
    int convert_to_int() const { return static_cast<int>(v_); }

private:
    double v_;
};
