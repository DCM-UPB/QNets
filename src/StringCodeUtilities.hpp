#ifndef STRING_CODE_UTILITIES
#define STRING_CODE_UTILITIES


#include <string>
#include <sstream>
#include <vector>

/*
--- Utilities for the string codes of StringCodeComponent ---

Types of string codes:

basic:
  idCode             -   a short identifier for object type, e.g. "nnu" for NeuralNetworkUnit.
  params             -   a space and comma separated list string of "name value" pairs, e.g. "i 10 , f 0.3 , b 1" for parameters of basic datatype.
  fullCode           -   "idCode ( params )", where idCode and params are like above

memberTreeCode:
  memberTreeIdCode   -   a list string of member ids, composed recursively. E.g. "a , b { c }", where a and b are members of the class and c is a member of b
  memberTreeFullCode -   like memberTreeIdCode, but with params included, e.g. "a ( i 10 ) , b { c ( f 0.3 ) }" , where a and c have a parameter and b has none

treeCode:
  treeIdCode         -   "idCode { memberTreeIdCode }"
  treeFullCode       -   "fullCode { memberFullCode }"

NOTE 1: The idCode identifiers should be uniquely identify a certain type among all derived types of StringCodeComponent.
NOTE 2: However you may have multiple codes of the same type / identifier in a list (then access via identifier will always yield the first appearance, so use index method version instead).
NOTE 3: The parameter identifiers of a class and its' childs must be unique, so every params code list will have unique element identifiers.
*/

// --- Readers

std::string readIdCode(const std::string &fullCode); // read idCode string from (tree)fullCode (i.e. passing a treeFullCode is also legit)
std::string readParams(const std::string &fullCode); // read params string from (tree)fullCode
std::string readParamValue(const std::string &params, const std::string &paramIdCode); // return the value string of certain paramId
std::string readMemberTreeCode(const std::string &treeCode); // return a string composed of the tree(Id/Full)Codes of all members in tree(Id/Full)Code
std::string readTreeCode(const std::string &memberTreeCode, const std::string &memberIdCode); // return the tree(Id/Full)Code of the first member with matching memberIdCode identifier from memberTree(Id/Full)Code
std::string readTreeCode(const std::string &memberTreeCode, const std::string &memberIdCode, const int &index); // return the tree(Id/Full)Code of the '(index-1)'th member with matching memberIdCode identifier from memberTree(Id/Full)Code

// --- Counters

int countNParams(const std::string &params); // count number of params in params string
int countTreeNParams(const std::string &treeCode); // count total number of params in treeCode string
int countNMembers(const std::string &memberTreeCode, const bool direct_only = true); // count number of direct (or total if direct_only == false) members in memberTreeCode string

// --- Composers

std::string composeCodes(const std::string &code1, const std::string &code2); // compose a string of the two codes separated by spaces and comma
std::string composeCodeList(const std::vector<std::string> &codes); // compose a string of vector elements separated by spaces and comma
std::string composeFullCode(const std::string &idCode, const std::string &params); // compose fullCode string from idCode and params
std::string composeTreeCode(const std::string &code, const std::string &memberTreeCode); // compose tree(Id/Full)Code string from (id/full)Code and memberTree(Id/Full)Code


// --- Templates

// for applying parameter value string to actual parameter, from param value string
template <typename T>
bool setParamValue(const std::string &paramValue, T &var)
{
    if (paramValue != "") {std::istringstream iss(paramValue); return !(iss >> var).fail();}
    else return false;
}

// for applying parameter value string to actual parameter, from full params list
template <typename T>
bool setParamValue(const std::string &params, const std::string &paramIdCode, T &var)
{
    return setParamValue(readParamValue(params, paramIdCode), var);
}

// for creating parameter value string from actual parameter
template <typename T>
std::string composeParamValue(const T &var)
{
    std::ostringstream oss;
    if (!(oss << var).fail()) return oss.str();
    else return "";
}

// for creating "name value" string from identifier and actual parameter
template <typename T>
std::string composeParamCode(const std::string &paramIdCode, const T &var)
{
    std::string value = composeParamValue(var);
    if (value != "") return paramIdCode + " " + value;
    else return "";
}

#endif

