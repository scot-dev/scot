# Maintainer: SCoT Development Team

pkgname=python-scot-git
pkgver=0.0.0
pkgrel=1
pkgdesc="EEG Source Connectivity Toolbox"
arch=('any')
url="https://github.com/SCoT-dev/SCoT"
license=('MIT')
provides=('python-scot')
makedepends=('git')
depends=('python' 'python-numpy' 'python-scipy')
optdepends=('python-scikit-learn: processing backend'
            'python-matplotlib: plotting support')
source=("$pkgname"::'git+https://github.com/SCoT-dev/SCoT.git')
md5sums=('SKIP')

pkgver() {
  cd "$srcdir/$pkgname"
  printf "%sr%s.%s" "$(cat VERSION)" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

build() {
  cd "$srcdir/$pkgname"
  # nothing to build - package is pure python
  #python setup.py build
}

package() {
  cd "$srcdir/$pkgname"
  python setup.py install --prefix=$pkgdir/usr --optimize=1
}
