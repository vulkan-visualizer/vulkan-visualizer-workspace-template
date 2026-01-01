cmake_minimum_required(VERSION 3.28)
project(vulkan-visualizer-workspace-template VERSION 1.2.0 LANGUAGES CXX)

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


add_subdirectory(vulkan-visualizer)

include(CMakeParseArguments)
function(add_slang_shader_target)
    find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

    cmake_parse_arguments(
            SLANG
            ""
            "TARGET;OUTPUT_DIR"
            "SOURCES"
            ${ARGN}
    )

    if (NOT SLANG_TARGET)
        message(FATAL_ERROR "add_slang_shader_target: TARGET is required")
    endif ()

    if (NOT SLANG_SOURCES)
        message(FATAL_ERROR "add_slang_shader_target: SOURCES is required")
    endif ()

    if (NOT SLANG_OUTPUT_DIR)
        set(SLANG_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    endif ()

    set(OUTPUTS)

    foreach (SRC ${SLANG_SOURCES})
        get_filename_component(SHADER_NAME ${SRC} NAME_WE)
        set(SPV ${SLANG_OUTPUT_DIR}/${SHADER_NAME}.spv)
        set(ENTRY_POINTS -entry vertMain -entry fragMain)
        add_custom_command(
                OUTPUT ${SPV}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${SLANG_OUTPUT_DIR}
                COMMAND ${SLANGC_EXECUTABLE}
                ${SRC}
                -target spirv
                -profile spirv_1_4
                -emit-spirv-directly
                -fvk-use-entrypoint-name
                ${ENTRY_POINTS}
                -o ${SPV}
                DEPENDS ${SRC}
                COMMENT "Compiling Slang shader: ${SRC}"
                VERBATIM
        )

        list(APPEND OUTPUTS ${SPV})
    endforeach ()

    add_custom_target(${SLANG_TARGET} ALL
            DEPENDS ${OUTPUTS}
    )

    set_target_properties(${SLANG_TARGET}
            PROPERTIES
            FOLDER "Shaders"
    )
endfunction()

file(GLOB SHADER_SLANG_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang
)

add_slang_shader_target(
        TARGET compile-slang-shaders
        SOURCES ${SHADER_SLANG_SOURCES}
)

add_executable(example-app)
target_sources(example-app
        PRIVATE
        example.app.cpp
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        example.app.ixx
)
target_link_libraries(example-app PRIVATE vk-core::vk-core)
add_dependencies(example-app compile-slang-shaders)