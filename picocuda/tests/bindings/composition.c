int h() {
    return 11;
}

int g() {
    return 10 + h();
}

int f() {
    return 9 + g();
}

int main() {
    return f();
}