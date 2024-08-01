
apt install libcgal-dev cmake

cd eval/uniformity/
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

