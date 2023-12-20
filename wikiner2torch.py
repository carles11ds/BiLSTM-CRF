import json

filename = "c:\\Users\\CarlesDuráSantonja\\Downloads\\aij-wikiner-en-wp3\\aij-wikiner-en-wp3"
with open(filename+".json", "w") as w:
    with open(filename, "r", encoding='utf-8') as f:
        cont = 0
        json_dict = {}
        for line in f.readlines():
            if line.strip():
                json_dict[str(cont)] = {"sentence": [], "labels": []}
                for token in line.split(" "):
                    tok = token.split("|")[0]
                    tag = token.split("|")[2]
                    json_dict[str(cont)]["sentence"].append(tok)
                    if json_dict[str(cont)]["labels"]:
                        last = json_dict[str(cont)]["labels"][-1]
                        if tag.strip() != "O":
                            if last.split("-")[-1] == tag.split("-")[-1].strip():
                                json_dict[str(cont)]["labels"].append("I-"+tag.split("-")[-1].strip())
                            else:
                                json_dict[str(cont)]["labels"].append("B-"+tag.split("-")[-1].strip())
                        else:
                            json_dict[str(cont)]["labels"].append(tag.strip())
                    else:
                        if tag.strip() != "O":
                            json_dict[str(cont)]["labels"].append("B-"+tag.split("-")[-1].strip())
                        else:
                            json_dict[str(cont)]["labels"].append(tag.strip())
            cont+=1
        json_data = json.dumps(json_dict, indent=2)
        w.write(json_data)
