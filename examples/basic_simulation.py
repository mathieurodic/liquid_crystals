from liquid_crystals import LC

def main():
    lc = LC(width=400, height=400, radius=2)
    lc.add_random_fields(10)
    lc.simulate(threaded=True)
    lc.display(cmap='special')
    
if __name__ == '__main__':
    main()
