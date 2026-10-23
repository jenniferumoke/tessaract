class UnsupportedToolOptionError(ValueError):
    '''A canonical tool option has no equivalent on the target provider.

        Tessaract raises instead of silently dropping the option, so the request never runs
        with settings the caller didn't get.
    '''
