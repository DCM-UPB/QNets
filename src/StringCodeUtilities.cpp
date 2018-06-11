#include "StringCodeUtilities.hpp"

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


std::string readMemberTreeFullCode(const std::string &treeFullCode)
{
    std::istringstream iss(treeFullCode);
    std::string word;
    std::string memberTreeFullCode = "";
    int countOpenBrackets = 0; // count total open { brackets

    iss >> word; // skip id
    iss >> word; // skip (
    while (iss >> word) { // search for )
        if (word == ")") break;
    }
    iss >> word; // skip {
    countOpenBrackets += 1;
    while (iss >> word) { // read in membersTreeFullCode
        if (word == "{") countOpenBrackets += 1;
        if (word == "}") countOpenBrackets -= 1;
        if (countOpenBrackets == 0) return memberTreeFullCode;
        if (memberTreeFullCode != "") memberTreeFullCode += " "; // add spacing
        memberTreeFullCode += word; // by placing it here the final } wont be added
    }
    return "";
}


std::string readTreeFullCode(const std::string &memberTreeFullCode, const std::string &memberIdCode)
{
    std::istringstream iss(memberTreeFullCode);
    std::string word;
    std::string treeFullCode = "";
    int countLeftBrackets = 0; // count { brackets of member code
    int countRightBrackets = 0; // count } brackets of member code

    while (iss >> word) { // search for memberIdCode
        if (word == memberIdCode) {
            treeFullCode = word; // read IdCode
            while (iss >> word) { // read rest
                treeFullCode += " " + word;
                if (word == "{") countLeftBrackets += 1;
                if (word == "}") countRightBrackets += 1;
                if (countLeftBrackets > 0 && countLeftBrackets == countRightBrackets) return treeFullCode;
            }
        }
    }

    return "";
}

// --- Writers

// compose fullCode string from idCode and params
std::string writeFullCode(const std::string &idCode, const std::string &params)
{
    return idCode + " ( " + params + " )";
}

// compose treeIdCode string from idCode and memberTreeIdCode
std::string writeTreeIdCode(const std::string &idCode, const std::string &memberTreeIdCode)
{
    return idCode + " { " + memberTreeIdCode + " }";
}

// compose treeFullCode string from fullCode and memberTreeFullCode
std::string writeTreeFullCode(const std::string &fullCode, const std::string &memberTreeFullCode)
{
    return fullCode + " { " + memberTreeFullCode + " }";
}
