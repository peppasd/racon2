cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make
sudo make install
cd ..
make
echo "done"
./test4
