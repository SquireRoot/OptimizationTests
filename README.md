# OptimizationTests

To run the tests, clone the repo

```
git clone git@github.com:SquireRoot/OptimizationTests.git
cd OptimizationTests/
```

This project depends on blis, eigen, and fftw. If you are on a linux system, run the following script to build and install all dependencies to the ext/ folder
```
./BuildDependencies.sh
```

Create a build Directory
```
mkdir build/ && cd build/
```

Build the project
```
cmake ../
make
```

Run the project
```
./OptimizationTests
```
