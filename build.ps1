
param (
    [string]$mode = ""
)

if ($mode -eq "all") {
    # Remove build directory if it exists
    if (Test-Path .\build) {
        Remove-Item -Recurse -Force .\build
    }

    # Create and enter build directory
    mkdir .\build
    Set-Location .\build

    # Run CMake configuration
    cmake .. -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug
} else {
    # If not "all", just go into build
    Set-Location .\build
}

# Build the project
cmake --build .

# Go to Debug directory and run the executable
Set-Location .\Debug
.\nanoGrad.exe
Set-Location .\..\..