#include <stdio.h>
#include <stdlib.h>

typedef char* string;

int compareFiles(string file_name1, string file_name2) {
  FILE *f1 = fopen(file_name1, "r");
  FILE *f2 = fopen(file_name2, "r");
  int ans = 1;
  char c1, c2;
  if (f1 && f2) {
    while ((c1 = fgetc(f1)) != EOF && (c2 = fgetc(f2)) != EOF) {
      if (c1 != c2) {
        ans = 0;
        break;
      }
    }
  } else {
    printf("Error opening files!\n");
    ans = 0;
  }
  fclose(f1);
  fclose(f2);
  return ans;
}

int main(int argc, char** argv) {
  if (argc =! 3) {
    printf("Must be called with the name of the two files\n");
    return 1;
  }
  string file_name1 = argv[1];
  string file_name2 = argv[2];
  printf("File names: %s, %s\n", file_name1, file_name2);
  if (compareFiles(file_name1, file_name2)) printf("The files are equal\n");
  else printf("The files are not equal\n");
  return 0;
}
