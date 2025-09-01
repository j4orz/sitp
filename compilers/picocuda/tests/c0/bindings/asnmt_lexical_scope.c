int main() {
    int x = 9;
    { int y = x*x*x; }
    return x;
}