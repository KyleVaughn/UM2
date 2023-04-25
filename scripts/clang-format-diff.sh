git ls-files -- '*.h' '*.hpp' '*.inl' '*.cpp' | xargs clang-format -i -style=file
git diff --exit-code --color
