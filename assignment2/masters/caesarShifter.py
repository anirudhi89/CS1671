import random

def caesarShift(message: str, shift: int=4, encrypt: bool =True):
    """
    message: string of the message you want encrypted
    shift: int number of how much each letter in the alphabet (a-z) should be shifted
    encrypt: boolean indicating if we are encrypting the message or decrypting the message all this changes is if we add the shift or subtract
    """
    #normalize shift to be between 0 and 25
    shift = shift%26
    print(shift)
    shift *= 1 if encrypt else -1
    retval = ""
    for i in range(len(message)):
        ch = message[i]
        if ch.isalpha():
            #we are a character between a-z or A-Z
            base = 65 if ch.isupper() else 97
            retval += chr((ord(ch) + shift - base) % 26 + base)
        else:
            retval += ch
    return retval

if __name__ == "__main__":
    exampleMessage = "The Quick Brown Fox jumps over the Lazy dog" # Original Message
    exampleEncrypt = caesarShift(exampleMessage, 23) # Encrypted Message and key of 23
    print(exampleMessage)
    print(exampleEncrypt)
    print(caesarShift(exampleEncrypt, 23, False))



