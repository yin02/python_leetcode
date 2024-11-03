```cpp
#include <iostream>//input output
using namespace std; //std::

int main() {
  cout << "Hello World!";// see out
  return 0; //end 
}
```

### example explained

- **line 1:** `#include <iostream>` is a header file library that lets us work with input and output objects, such as `cout` (used in line 5). header files add functionality to c++ programs.
  
- **line 2:** `using namespace std` means that we can use names for objects and variables from the standard library.

  *don't worry if you don't understand how `#include <iostream>` and `using namespace std` works. just think of it as something that (almost) always appears in your program.*

- **line 3:** a blank line. c++ ignores white space, but we use it to make the code more readable.

- **line 4:** another thing that always appears in a c++ program is `int main()`. this is called a function. any code inside its curly brackets `{}` will be executed.

- **line 5:** `cout` (pronounced "see-out") is an object used together with the insertion operator `<<` to output/print text. in our example, it will output `"Hello World!"`.

  - *note:* c++ is case-sensitive: `cout` and `Cout` have different meanings.
  - *note:* every c++ statement ends with a semicolon `;`.
  - *note:* the body of `int main()` could also be written as:
    ```cpp
    int main () { cout << "Hello World! "; return 0; }
    ```
  
  *remember: the compiler ignores white spaces. however, multiple lines make the code more readable.*

- **line 6:** `return 0;` ends the main function.

- **line 7:** do not forget to add the closing curly bracket `}` to actually end the main function.

### omitting namespace

you might see some c++ programs that run without the standard namespace library. the `using namespace std` line can be omitted and replaced with the `std` keyword, followed by the `::` operator for some objects:

#### example
```cpp
#include <iostream>

int main() {
  std::cout << "Hello World!";
  return 0;
}
```




```cpp
#include <iostream>
using namespace std;

int main() {
  int x;
  cout << "Type a number: "; // Type a number and press enter
  cin >> x; // Get user input from the keyboard
  cout << "Your number is: " << x;
  return 0;
}

```


https://www.w3schools.com/cpp/cpp_user_input.asp