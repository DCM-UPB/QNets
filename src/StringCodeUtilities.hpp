#ifndef STRING_CODE_UTILITIES
#define STRING_CODE_UTILITIES


#include <string>
#include <sstream>
#include <vector>

/*
--- Utilities for the string codes of StringCodeComponent ---
Types of string codes:

basic:
  idCode             -   a short unique identifier for object type
  params             -   a space separated string of "name value" pairs, e.g. "i 10 f 0.3 b 1" for parameters of basic datatype
  fullCode           -   "idCode ( params )", where idCode and params are like above

memberTreeCode:
  memberTreeIdCode   -   a string of member ids, composed recursively. E.g. "a { } b { c { } }", where a and b are members of the class and c is a member of b
  memberTreeFullCode -   like memberTreeIdCode, but with params included, e.g. "a ( i 10 ) b ( ) { c ( f 0.3 ) { } }"

treeCode:
  treeIdCode         -   "idCode { memberTreeIdCode }"
  treeFullCode       -   "fullCode { memberFullCode }"
*/

// --- Readers

std::string readIdCode(const std::string &fullCode); // read idCode string from (tree)fullCode (i.e. passing a treeFullCode is also legit)
std::string readParams(const std::string &fullCode); // read params string from (tree)fullCode
std::string readParamValue(const std::string &params, const std::string &paramIdCode); // return the value string of certain paramId
std::string readMemberTreeCode(const std::string &treeCode); // return a string composed of the tree(Id/Full)Codes of all members in tree(Id/Full)Code
std::string readTreeCode(const std::string &memberTreeCode, const std::string &memberIdCode); // return the tree(Id/Full)Code of a certain member from memberTree(Id/Full)Code

/* not yet implemented
int readNParams(const std::string &params); // count number of params in params string
int readTreeNParams(const std::string &treeCode); // count total number of params in treeCode string
int readNMembers(const std::string &memberTreeCode); // count number of direct members in memberTreeCode string
int readTreeNMembers(const std::string &memberTreeCode); // count number of direct and indirect members in tree string
*/

// --- Composers

std::string composeCodes(const std::vector<std::string> &codes); // compose a string of elements separated by spaces from a vector of codes
std::string composeFullCode(const std::string &idCode, const std::string &params); // compose fullCode string from idCode and params
std::string composeTreeCode(const std::string &code, const std::string &memberTreeCode); // compose tree(Id/Full)Code string from (id/full)Code and memberTree(Id/Full)Code


// --- Setter Template

// for applying parameter value string to actual parameter
template <typename T>
bool setParamValue(const std::string &params, const std::string &paramIdCode, T &var)
{
    std::string ret = readParamValue(params, paramIdCode);
    if (ret!="") {std::istringstream iss(ret); return !(iss >> var).fail();}
    else return false;
}

#endif

