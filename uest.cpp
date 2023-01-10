#include <fstream>
#include <string>
#include <iostream>

int main() {
  std::ifstream file("test/data/window.txt");
  std::string str;

    std::getline(file, str);
    int readnum = std::stoi(str);

    while(readnum > 0){
        std::getline(file, str);
        std::cout << str << std::endl;
        readnum --;
    }
  

  return 0;
}
