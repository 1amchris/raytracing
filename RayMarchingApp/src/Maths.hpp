#pragma once

#include <cstring> 
#include <memory>
#include <iostream>

#include "Hardware.hpp"


namespace Utils 
{
    void SwapMemory(void* address1, void* address2, size_t size) 
    {
        void* temp = std::malloc(size);
        
        if (temp == nullptr) 
        {
            std::cerr << "Memory allocation failed. Couldn't swap memory in provided addresses." << std::endl;
            return;
        }

        std::memcpy(temp, address1, size);
        std::memcpy(address1, address2, size);
        std::memcpy(address2, temp, size);

        std::free(temp);
    }
}