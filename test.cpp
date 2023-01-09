#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <mutex>

std::mutex lock_table[5];

void stage1(std::istream& in, std::ostream& out)
{
    int n;
    while (in >> n) {
        std::lock_guard<std::mutex> lock(lock_table[n]);
        out << n * n << ' ';
    }
}

void stage2(std::istream& in, std::ostream& out)
{
    int n;
    while (in >> n) {
        std::lock_guard<std::mutex> lock(lock_table[n]);
        out << n + 1 << ' ';
    }
}

int main()
{
    std::stringstream input;
    std::stringstream output1;
    std::stringstream output2;

    std::vector<int> data{1, 2, 3, 4, 5};
    for (int n : data) {
        input << n << ' ';
    }

    std::thread t1(stage1, std::ref(input), std::ref(output1));
    std::thread t2(stage2, std::ref(output1), std::ref(output2));

    t1.join();
    t2.join();

    std::string result = output2.str();
    std::cout << result << std::endl;

    return 0;
}
