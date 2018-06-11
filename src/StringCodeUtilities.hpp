#ifndef STRING_CODE_UTILITIES
#define STRING_CODE_UTILITIES


#include <string>
#include <vector>

// --- Readers

std::string readIdCode(const std::string &fullCode); // read idCode string from fullCode (passing treeFullCode is also legit)
std::string readParams(const std::string &fullCode); // read params string from fullCode (passing treeFullCode is also legit)
std::string readParamValue(const std::string &fullCode, const std::string &paramIdCode); // return the value string of certain paramId
std::string readMemberTreeFullCode(const std::string treeFullCode); // return a string composed of the treeFullCodes of all members
std::string readTreeFullCode(const std::string memberTreeFullCode, const std::string memberIdCode); // return the treeFullCode of a certain member

// --- Writers

std::string writeFullCode(const std::string &idCode, const std::string &params); // compose fullCode string from idCode and params
std::string writeTreeIdCode(const std::string &idCode, const std::string &memberTreeIdCode); // compose treeIdCode string from idCode and memberTreeIdCode
std::string writeTreeFullCode(const std::string &fullCode, const std::string &memberTreeFullCode); // compose treeFullCode string from fullCode and memberTreeFullCode


#endif

