set(EXE_TARGET_NAME reaching)

set(${EXE_TARGET_NAME}_SRC
        main.cpp
)

add_executable(${EXE_TARGET_NAME} ${${EXE_TARGET_NAME}_SRC})

target_link_libraries(${EXE_TARGET_NAME})

install(TARGETS ${EXE_TARGET_NAME} DESTINATION bin)
