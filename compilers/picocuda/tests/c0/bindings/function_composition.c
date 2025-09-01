int h() {
    return 11;
}

int g() {
    return 10 + h();
}

int f(int x, int y) {
    return x + y + g();
}

int main() {
    return f(9, 10) + 3;
}