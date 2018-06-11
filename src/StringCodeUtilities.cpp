#include "StringCodeUtilities.hpp"

#include <vector>
#include <string>
#include <sstream>

// --- Readers

std::string readIdCode(const std::string &fullCode)
{
    std::istringstream iss(fullCode);
    std::string word = "";
    iss >> word;
    return word;
}


std::string readParams(const std::string &fullCode)
{
    std::istringstream iss(fullCode);
    std::string word;
    std::string params = "";
    iss >> word; // skip id
    iss >> word; // skip (
    while (iss >> word) { // read params string
        if (word == ")") return params; // don't add the )
        if (params != "") params += " "; // add spacing
        params += word;
    }
    return "";
}


std::string readParamValue(const std::string &params, const std::string &paramIdCode)
{
    std::istringstream iss(params);
    std::string word;
    while (iss >> word) { // search for paramIdCode
        if (word == paramIdCode) {
            iss >> word; // read value
            return word;
        }
    }
    return "";
}


std::string readMemberTreeCode(const std::string &treeCode)
{
    std::istringstream iss(treeCode);
    std::string word;
    std::string memberTreeCode = "";
    int countOpenBrackets = 0; // count total open { brackets

    iss >> word; // skip id
    iss >> word; // skip ( or {
    if (word == "(") { // if treeCode is treeFullCode we need to skip params bracket
        while (iss >> word) { // search for )
            if (word == ")") break;
        }
        iss >> word; // skip {
    }
    countOpenBrackets += 1;
    while (iss >> word) { // read in membersTreeFullCode
        if (word == "{") countOpenBrackets += 1;
        if (word == "}") countOpenBrackets -= 1;
        if (countOpenBrackets == 0) return memberTreeCode;
        if (memberTreeCode != "") memberTreeCode += " "; // add spacing
        memberTreeCode += word; // by placing it here the final } wont be added
    }
    return "";
}


std::string readTreeCode(const std::string &memberTreeCode, const std::string &memberIdCode)
{
    std::istringstream iss(memberTreeCode);
    std::string word;
    std::string treeCode = "";
    int countLeftBrackets = 0; // count { brackets of member code
    int countRightBrackets = 0; // count } brackets of member code

    while (iss >> word) { // search for memberIdCode
        if (word == memberIdCode) {
            treeCode = word; // read IdCode
            while (iss >> word) { // read rest
                treeCode += " " + word;
                if (word == "{") countLeftBrackets += 1;
                if (word == "}") countRightBrackets += 1;
                if (countLeftBrackets > 0 && countLeftBrackets == countRightBrackets) return treeCode;
            }
        }
    }

    return "";
}

// --- Composers

std::string composeCodes(const std::vector<std::string> &codes)
{
    if (codes.size() < 1) return ""; // to be safe
    std::string composedCode = codes[0]; // start with first code
    for (std::vector<std::string>::size_type i=1; i<codes.size(); ++i) composedCode += " " + codes[i]; // append other codes with spacing
    return composedCode;
}


std::string composeFullCode(const std::string &idCode, const std::string &params)
{
    return idCode + " ( " + params + " )";
}


std::string composeTreeCode(const std::string &code, const std::string &memberTreeCode)
{
    return code + " { " + memberTreeCode + " }";
}
