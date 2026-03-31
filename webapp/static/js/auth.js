/**
 * auth.js - Client-side JWT authentication helper.
 *
 * Handles:
 *  1. Redirect to /login if no token is stored and page requires auth.
 *  2. Patching fetch() / jQuery $.ajax() to inject the Authorization header.
 *  3. Logout logic (clear token, redirect).
 *  4. Updating the navbar to show Login vs user email + Logout.
 */

(function () {
    'use strict';

    /* ------------------------------------------------------------------ */
    /*  Helpers                                                            */
    /* ------------------------------------------------------------------ */

    var TOKEN_KEY = 'access_token';
    var EMAIL_KEY = 'user_email';

    function getToken() {
        return localStorage.getItem(TOKEN_KEY);
    }

    function isLoggedIn() {
        return !!getToken();
    }

    function logout() {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(EMAIL_KEY);
        window.location.href = '/login';
    }

    /* ------------------------------------------------------------------ */
    /*  Redirect guard – pages that need auth redirect to /login           */
    /* ------------------------------------------------------------------ */

    // Pages that anyone can see without logging in.
    var publicPaths = ['/', '/login', '/about', '/results', '/predict_page'];

    function isPublicPage() {
        var path = window.location.pathname.replace(/\/+$/, '') || '/';
        for (var i = 0; i < publicPaths.length; i++) {
            if (path === publicPaths[i]) return true;
        }
        return false;
    }

    // If the user is NOT logged in and the page is protected, redirect.
    if (!isLoggedIn() && !isPublicPage()) {
        window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
        // Stop executing the rest of the script.
        return;
    }

    /* ------------------------------------------------------------------ */
    /*  Patch window.fetch to auto-attach the Bearer token                 */
    /* ------------------------------------------------------------------ */

    var _originalFetch = window.fetch;

    window.fetch = function (input, init) {
        init = init || {};

        // Only attach for same-origin /api/ requests.
        var url = typeof input === 'string' ? input : (input && input.url ? input.url : '');
        if (url.indexOf('/api/') !== -1 && getToken()) {
            init.headers = init.headers || {};
            // Support both Headers object and plain object.
            if (init.headers instanceof Headers) {
                if (!init.headers.has('Authorization')) {
                    init.headers.set('Authorization', 'Bearer ' + getToken());
                }
            } else {
                if (!init.headers['Authorization']) {
                    init.headers['Authorization'] = 'Bearer ' + getToken();
                }
            }
        }

        return _originalFetch.call(this, input, init).then(function (response) {
            // If API returns 401, token is invalid/expired – log the user out.
            if (response.status === 401 && url.indexOf('/api/auth/login') === -1 && url.indexOf('/api/auth/register') === -1) {
                logout();
            }
            return response;
        });
    };

    /* ------------------------------------------------------------------ */
    /*  Patch jQuery $.ajax (if jQuery is loaded later) to attach token     */
    /* ------------------------------------------------------------------ */

    function patchJQueryIfLoaded() {
        if (typeof $ !== 'undefined' && $.ajaxSetup) {
            $.ajaxSetup({
                beforeSend: function (xhr, settings) {
                    if (settings.url && settings.url.indexOf('/api/') !== -1 && getToken()) {
                        xhr.setRequestHeader('Authorization', 'Bearer ' + getToken());
                    }
                },
                statusCode: {
                    401: function () {
                        if (window.location.pathname !== '/login') {
                            logout();
                        }
                    }
                }
            });
        }
    }

    // Patch now if jQuery exists, or wait for DOMContentLoaded
    patchJQueryIfLoaded();
    document.addEventListener('DOMContentLoaded', patchJQueryIfLoaded);

    /* ------------------------------------------------------------------ */
    /*  Navbar – swap Login / Logout button                                */
    /* ------------------------------------------------------------------ */

    document.addEventListener('DOMContentLoaded', function () {
        var authNav = document.getElementById('authNavItem');
        if (!authNav) return;

        if (isLoggedIn()) {
            var email = localStorage.getItem(EMAIL_KEY) || 'Account';
            // Extract name portion of email for display
            var displayName = email.split('@')[0];
            if (displayName.length > 16) displayName = displayName.substring(0, 14) + '..';

            authNav.innerHTML =
                '<div class="d-flex align-items-center gap-2">' +
                '  <div class="nav-user-badge">' +
                '    <span class="nav-avatar">' + displayName.charAt(0).toUpperCase() + '</span>' +
                '    <span class="nav-user-name">' + displayName + '</span>' +
                '  </div>' +
                '  <a class="nav-logout-btn" href="#" id="logoutBtn" title="Sign out">' +
                '    <i class="fas fa-sign-out-alt" style="margin:0"></i>' +
                '  </a>' +
                '</div>';

            document.getElementById('logoutBtn').addEventListener('click', function (e) {
                e.preventDefault();
                logout();
            });
        } else {
            authNav.innerHTML =
                '<a class="nav-login-btn" href="/login"><i class="fas fa-sign-in-alt" style="margin:0;margin-right:6px"></i>Sign In</a>';
        }
    });

    /* ------------------------------------------------------------------ */
    /*  Expose helpers globally so other scripts can use them               */
    /* ------------------------------------------------------------------ */
    window.AuthHelper = {
        getToken: getToken,
        isLoggedIn: isLoggedIn,
        logout: logout
    };
})();
