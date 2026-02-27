import traceback
import sys
try:
    import main
    main.main()
except Exception as e:
    with open('err.txt', 'w') as f:
        traceback.print_exc(file=f)
