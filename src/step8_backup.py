import sys, shutil, glob, os, datetime
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = ["models/step6_final.joblib", "data/raw/emails.csv"] + glob.glob("data/processed/*.csv")

def backup():
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    for f in FILES:
        if os.path.exists(f):
            os.makedirs(f"backups/{stamp}/{os.path.dirname(f)}", exist_ok=True)
            shutil.copy2(f, f"backups/{stamp}/{f}")
    open("backups/CURRENT.txt", "w").write(stamp)
    for d in sorted(x for x in os.listdir("backups") if x[:2] == "20")[:-10]:
        shutil.rmtree(f"backups/{d}")   # keep the newest 10 backups
    return stamp

def rollback(stamp):
    for f in FILES:
        if os.path.exists(f"backups/{stamp}/{f}"):
            shutil.copy2(f"backups/{stamp}/{f}", f)
    open("backups/CURRENT.txt", "w").write(stamp)
    print("rolled back to", stamp)

if __name__ == "__main__":
    os.makedirs("backups", exist_ok=True)
    if "--list" in sys.argv:
        print(sorted(x for x in os.listdir("backups") if x[:2] == "20"))
    elif "--rollback" in sys.argv:
        rollback(sys.argv[sys.argv.index("--rollback") + 1])
    else:
        print("backup saved:", backup())
