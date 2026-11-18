Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=97242bf8a40f049f ===
Gold sentences and facts:
Sentence 0: Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name ( Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741 7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743 -- First Lord of the Treasury Whig Carteret 7002502000000000000 ♠ 1 year and 137 days ​ The Right Honourable Henry Pelham FRS MP for Sussex ( 1694 -- 1754 ) 27 August 1743 6 March 1754 1747 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Broad Bottom I Broad Bottom II 7003384500000000000 ♠ 10 years and 192 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 16 March 1754 16 November 1756 1754 First Lord of the Treasury Leader of the House of Lords Whig Newcastle I 7002977000000000000 ♠ 2 years and 246 days His Grace William Cavendish 4th Duke of Devonshire KG PC ( 1720 -- 1764 ) 16 November 1756 25 June 1757 -- First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Devonshire ​ 1757 Caretaker 7002222000000000000 ♠ 222 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 2 July 1757 26 May 1762 1761 First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Newcastle George III ( 1760 -- 1820 ) 7003179000000000000 ♠ 4 years and 329 days The Right Honourable John Stuart 3rd Earl of Bute KG PC ( 1713 -- 1792 ) 26 May 1762 8 April 1763 -- First Lord of the Treasury Leader of the House of Lords Tory Bute ( Tory -- Whig ) 7002318000000000000 ♠ 318 days The Right Honourable George Grenville MP for Buckingham ( 1712 -- 1770 ) 16 April 1763 13 July 1765 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig ( Grenvillite ) Grenville ( Whig -- Tory ) 7002820000000000000 ♠ 2 years and 89 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 13 July 1765 30 July 1766 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham I 7002383000000000000 ♠ 1 year and 18 days The Right Honourable William Pitt 1st Earl of Chatham PC FRS MP for Bath ( 1708 -- 1778 ) 30 July 1766 14 October 1768 1768 Lord Keeper of the Privy Seal Whig ( Chathamite ) Chatham ( Whig -- Tory ) 7002808000000000000 ♠ 2 years and 77 days His Grace Augustus FitzRoy 3rd Duke of Grafton KG PC ( 1735 -- 1811 ) 14 October 1768 28 January 1770 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Grafton 7002472000000000000 ♠ 1 year and 107 days The Right Honourable Frederick North Lord North KG MP for Banbury ( 1732 -- 1792 ) 28 January 1770 22 March 1782 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Northite ) North 1774 1780 7003443700000000000 ♠ 12 years and 54 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 27 March 1782 1 July 1782 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham II 7001970000000000000 ♠ 97 days The Right Honourable William Petty 2nd Earl of Shelburne KG PC ( 1737 -- 1805 ) 4 July 1782 2 April 1783 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Shelburne ( Whig -- Tory ) 7002273000000000000 ♠ 273 days His Grace William Cavendish - Bentinck 3rd Duke of Portland PC FRS ( 1738 -- 1809 ) 2 April 1783 19 December 1783 -- First Lord of the Treasury Leader of the House of Lords Whig Fox -- North 7002262000000000000 ♠ 262 days The Right Honourable William Pitt the Younger MP for Appleby -- Cambridge University ( 1759 -- 1806 ) 19 December 1783 14 March 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt I 1784 1790 1796 7003629500000000000 ♠ 17 years and 86 days The Right Honourable Henry Addington MP for Devizes ( 1757 -- 1844 ) 17 March 1801 10 May 1804 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Addington 1802 7003115100000000000 ♠ 3 years and 55 days The Right Honourable William Pitt the Younger MP for Cambridge University ( 1759 -- 1806 ) 10 May 1804 23 January 1806 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt II 7002624000000000000 ♠ 1 year and 259 days The Right Honourable William Grenville 1st Baron Grenville PC PC ( Ire ) ( 1759 -- 1834 ) 11 February 1806 31 March 1807 First Lord of the Treasury Leader of the House of Lords Whig All the Talents ( Whig -- Tory ) 1806 7002414000000000000 ♠ 1 year and 49 days His Grace William Cavendish - Bentinck 3rd Duke of Portland KG PC FRS ( 1738 -- 1809 ) 31 March 1807 4 October 1809 First Lord of the Treasury Tory ( Pittite ) Portland II 1807 7002919000000000000 ♠ 2 years and 188 days The Right Honourable Spencer Perceval KC MP for Northampton ( 1762 -- 1812 ) 4 October 1809 11 May 1812 -- First Lord of the Treasury Chancellor of the Exchequer Chancellor of the Duchy of Lancaster Leader of the House of Commons Tory ( Pittite ) Perceval 7002951000000000000 ♠ 2 years and 221 days The Right Honourable Robert Jenkinson 2nd Earl of Liverpool KG PC FRS ( 1770 -- 1828 ) 8 June 1812 9 April 1827 1812 First Lord of the Treasury Leader of the House of Lords Tory ( Pittite ) Liverpool 1818 1820 George IV ( 1820 -- 1830 ) 1826 7003541900000000000 ♠ 14 years and 306 days The Right Honourable George Canning FRS MP for Seaford ( 1770 -- 1827 ) 10 April 1827 8 August 1827 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Canningite ) Canning ( Canningite -- Whig ) 7002121000000000000 ♠ 121 days The Right Honourable Frederick John Robinson 1st Viscount Goderich PC ( 1782 -- 1859 ) 31 August 1827 21 January 1828 -- First Lord of the Treasury Leader of the House of Lords Tory ( Canningite ) Goderich 7002144000000000000 ♠ 144 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 22 January 1828 16 November 1830 First Lord of the Treasury Leader of the House of Lords Tory Wellington -- Peel 1830 William IV ( 1830 -- 1837 ) 7003103000000000000 ♠ 2 years and 299 days The Right Honourable Charles Grey 2nd Earl Grey KG PC ( 1764 -- 1845 ) 22 November 1830 9 July 1834 First Lord of the Treasury Leader of the House of Lords Whig Grey 1831 1832 7003132600000000000 ♠ 3 years and 230 days The Right Honourable William Lamb 2nd Viscount Melbourne PC ( 1779 -- 1848 ) 16 July 1834 14 November 1834 -- First Lord of the Treasury Leader of the House of Lords Whig Melbourne I 7002122000000000000 ♠ 122 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 14 November 1834 10 December 1834 -- First Lord of the Treasury Sec. of State for the Home Department Sec. of State for Foreign Affairs Sec. of State for War and the Colonies Leader of the House of Lords Tory Wellington Caretaker 7001270000000000000 ♠ 27 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 10 December 1834 8 April 1835 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Conservative Peel I 7002120000000000000 ♠ 120 days The Right Honourable William Lamb 2nd Viscount Melbourne PC FRS ( 1779 -- 1848 ) 18 April 1835 30 August 1841 1835 First Lord of the Treasury Leader of the House of Lords Whig Melbourne II 1837 Victoria ( 1837 -- 1901 ) 7003232700000000000 ♠ 6 years and 135 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 30 August 1841 29 June 1846 1841 First Lord of the Treasury Leader of the House of Commons Conservative Peel II 7003176500000000000 ♠ 4 years and 304 days The Right Honourable Lord John Russell FRS MP for the City of London ( 1792 -- 1878 ) 30 June 1846 21 February 1852 1847 First Lord of the Treasury Leader of the House of Commons Whig Russell I 7003206300000000000 ♠ 5 years and 237 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 23 February 1852 17 December 1852 First Lord of the Treasury Leader of the House of Lords Conservative Who ? Who ? 1852 7002299000000000000 ♠ 299 days The Right Honourable George Hamilton - Gordon 4th Earl of Aberdeen KT FRSE FRS PC FSA Scot ( 1784 -- 1860 ) 19 December 1852 30 January 1855 -- First Lord of the Treasury Leader of the House of Lords Peelite Aberdeen ( Peelite -- Whig ) 7002773000000000000 ♠ 2 years and 43 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 6 February 1855 19 February 1858 First Lord of the Treasury Leader of the House of Commons Whig Palmerston I 1857 7003111000000000000 ♠ 3 years and 14 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 20 February 1858 11 June 1859 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli II 7002477000000000000 ♠ 1 year and 112 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 12 June 1859 18 October 1865 1859 First Lord of the Treasury Leader of the House of Commons Liberal Palmerston II 1865 7003232100000000000 ♠ 6 years and 129 days The Right Honourable John Russell 1st Earl Russell KG PC FRS ( 1792 -- 1878 ) 29 October 1865 26 June 1866 -- First Lord of the Treasury Leader of the House of Lords Liberal Russell II 7002241000000000000 ♠ 241 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby KG PC ( 1799 -- 1869 ) 28 June 1866 25 February 1868 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli III 7002608000000000000 ♠ 1 year and 243 days The Right Honourable Benjamin Disraeli MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 -- 7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and 77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April 1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone II 23 April 1880 9 June 1885 1880 7003187400000000000 ♠ 5 years and 48 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 23 June 1885 28 January 1886 -- Sec. of State for Foreign Affairs Leader of the House of Lords Conservative Salisbury I 7002220000000000000 ♠ 220 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone III 1 February 1886 20 July 1886 1885 7002170000000000000 ♠ 170 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 July 1886 11 August 1892 1886 Leader of the House of Lords First Lord of the Treasury Sec. of State for Foreign Affairs Conservative Salisbury II 7003221000000000000 ♠ 6 years and 18 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone IV 15 August 1892 2 March 1894 1892 7002565000000000000 ♠ 1 year and 200 days The Right Honourable Archibald Primrose 5th Earl of Rosebery KG PC FRS ( 1847 -- 1929 ) 5 March 1894 22 June 1895 -- First Lord of the Treasury Lord President of the Council Leader of the House of Lords Liberal Rosebery 7002475000000000000 ♠ 1 year and 110 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 June 1895 11 July 1902 Leader of the House of Lords Sec. of State for Foreign Affairs Lord Keeper of the Privy Seal Conservative Salisbury III ( Con. -- Lib. U . ) 1895 1900 7003257300000000000 ♠ 7 years and 17 days Edward VII ( 1901 -- 1910 ) The Right Honourable Arthur Balfour FRS DL MP for Manchester East ( 1848 -- 1930 ) 11 July 1902 5 December 1905 -- First Lord of the Treasury Leader of the House of Commons Conservative Balfour 7003124400000000000 ♠ 3 years and 148 days The Right Honourable Sir Henry Campbell - Bannerman GCB MP for Stirling Burghs ( 1836 -- 1908 ) 5 December 1905 7 April 1908 First Lord of the Treasury Leader of the House of Commons Liberal Campbell - Bannerman 1906 7002855000000000000 ♠ 2 years and 125 days The Right Honourable Herbert Henry Asquith KC FRS MP for East Fife ( 1852 -- 1928 ) 7 April 1908 5 December 1916 First Lord of the Treasury Leader of the House of Commons Sec. of State for War Liberal Asquith I 1910 ​ 1910 George V ( 1910 -- 1936 ) ​ Asquith II ( Lib. -- Con. -- et al . ) 7003316500000000000 ♠ 8 years and 243 days The Right Honourable David Lloyd George OM MP for Caernarvon Boroughs ( 1863 -- 1945 ) 6 December 1916 19 October 1922 First Lord of the Treasury Liberal Lloyd George War 1918 Lloyd George II ( Lib. -- Con . ) 7003214400000000000 ♠ 5 years and 318 days The Right Honourable Bonar Law MP for Glasgow Central ( 1858 -- 1923 ) 23 October 1922 20 May 1923 1922 First Lord of the Treasury Leader of the House of Commons Conservative ( Scot. U . ) Law 7002210000000000000 ♠ 210 days The Right Honourable Stanley Baldwin JP MP for Bewdley ( 1867 -- 1947 ) 23 May 1923 16 January 1924 -- First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Conservative Baldwin I 7002239000000000000 ♠ 239 days The Right Honourable Ramsay MacDonald MP for Aberavon ( 1866 -- 1937 ) 22 January 1924 4 November 1924 1923 First Lord of the Treasury Leader of the House of Commons Sec. of State for Foreign Affairs Labour MacDonald I 7002288000000000000 ♠ 288 days The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 4 November 1924 5 June 1929 1924 First Lord of the Treasury Leader of the House of Commons Conservative Baldwin II 7003167500000000000 ♠ 4 years and 214 days The Right Honourable Ramsay MacDonald FRS MP for Seaham ( 1866 -- 1937 ) 5 June 1929 7 June 1935 1929 First Lord of the Treasury Leader of the House of Commons Labour MacDonald II National Labour National I ( N. Lab. -- Con. -- et al . ) 1931 National II 7003219400000000000 ♠ 6 years and 3 days ​ The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 7 June 1935 28 May 1937 1935 First Lord of the Treasury Leader of the House of Commons Conservative National III ( Con. -- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​ Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26 July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker ( Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26 July 1945 26 October 1951 1945 First Lord of the Treasury Minister of Defence Labour Attlee I 1950 Attlee II 7003228400000000000 ♠ 6 years and 93 days The Right Honourable Sir Winston Churchill KG OM CH TD PCc DL FRS RA MP for Woodford ( 1874 -- 1965 ) 26 October 1951 6 April 1955 1951 First Lord of the Treasury Minister of Defence Conservative Churchill III 7003125900000000000 ♠ 3 years and 163 days Elizabeth II ( 1952 -- present ) The Right Honourable Sir Anthony Eden KG MC MP for Warwick & Leamington ( 1897 -- 1977 ) 6 April 1955 10 January 1957 1955 First Lord of the Treasury Conservative Eden 7002646000000000000 ♠ 1 year and 280 days The Right Honourable Harold Macmillan FRS MP for Bromley ( 1894 -- 1986 ) 10 January 1957 19 October 1963 1959 First Lord of the Treasury Conservative Macmillan I ​ Macmillan II 7003247400000000000 ♠ 6 years and 283 days The Right Honourable Sir Alec Douglas - Home KT MP for Kinross & Western Perthshire ( 1903 -- 1995 ) 19 October 1963 16 October 1964 -- First Lord of the Treasury Conservative ( Scot. U . ) Douglas - Home 7002364000000000000 ♠ 364 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 16 October 1964 19 June 1970 1964 First Lord of the Treasury Minister for the Civil Service Labour Wilson I 1966 Wilson II 7003207300000000000 ♠ 5 years and 247 days The Right Honourable Edward Heath MBE MP for Bexley -- Sidcup ( 1916 -- 2005 ) 19 June 1970 4 March 1970 First Lord of the Treasury Minister for the Civil Service Conservative Heath 7003135500000000000 ♠ 3 years and 259 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 4 March 5 April 1976 First Lord of the Treasury Minister for the Civil Service Labour Wilson III Wilson IV 7002764000000000000 ♠ 2 years and 33 days The Right Honourable James Callaghan MP for Cardiff South East ( 1912 -- 2005 ) 5 April 1976 4 May 1979 -- First Lord of the Treasury Minister for the Civil Service Labour Callaghan 7003112500000000000 ♠ 3 years and 30 days The Right Honourable Margaret Thatcher FRS FRIC MP for Finchley ( 1925 -- 2013 ) See also § Premierships :3 First Lord of the Treasury Minister for the Civil Service Conservative Thatcher I 4 May 1979 28 November 1990 1979 Thatcher II Thatcher III 7003422700000000000 ♠ 11 years and 209 days The Right Honourable John Major MP for Huntingdon ( born 1943 ) 28 November 1990 2 May 1997 First Lord of the Treasury Minister for the Civil Service Conservative Major I ​ Major II 7003234800000000000 ♠ 6 years and 156 days The Right Honourable Tony Blair MP for Sedgefield ( born 1953 ) See also § Premierships : 4 First Lord of the Treasury Minister for the Civil Service Labour Blair I 2 May 1997 27 June 2007 1997 Blair II 2005 Blair III 7003370900000000000 ♠ 10 years and 57 days The Right Honourable Gordon Brown MP for Kirkcaldy & Cowdenbeath ( born 1951 ) See also § Premierships : 5 First Lord of the Treasury Minister for the Civil Service Labour Brown 27 June 2007 11 May -- 7003105000000000000 ♠ 2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July 2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days
  Fact 0.0: Robert Walpole, 1st Earl of Orford, was MP for King's Lynn and lived from 1676 to 1745.
  Fact 0.1: Robert Walpole was Prime Minister from 4 April 1721 to 11 February 1742, for 20 years and 314 days.
  Fact 0.2: Robert Walpole was a Whig.
  Fact 0.3: Spencer Compton, 1st Earl of Wilmington, lived from 1673 to 1743.
  Fact 0.4: Spencer Compton was Prime Minister from 16 February 1742 to 2 July 1743, for 1 year and 137 days.
  Fact 0.5: Spencer Compton was a Whig.
  Fact 0.6: Henry Pelham was MP for Sussex and lived from 1694 to 1754.
  Fact 0.7: Henry Pelham was Prime Minister from 27 August 1743 to 6 March 1754, for 10 years and 192 days.
  Fact 0.8: Henry Pelham was a Whig.
  Fact 0.9: Thomas Pelham-Holles, 1st Duke of Newcastle, lived from 1693 to 1768.
  Fact 0.10: Thomas Pelham-Holles was Prime Minister from 16 March 1754 to 16 November 1756, for 2 years and 246 days.
  Fact 0.11: Thomas Pelham-Holles was a Whig.
  Fact 0.12: William Cavendish, 4th Duke of Devonshire, lived from 1720 to 1764.
  Fact 0.13: William Cavendish was Prime Minister from 16 November 1756 to 25 June 1757, for 222 days.
  Fact 0.14: William Cavendish's ministry was a caretaker ministry in 1757.
  Fact 0.15: William Cavendish was a Whig.
  Fact 0.16: Thomas Pelham-Holles was Prime Minister again from 2 July 1757 to 26 May 1762, for 4 years and 329 days.
  Fact 0.17: Thomas Pelham-Holles was a Whig in his second term.
  Fact 0.18: John Stuart, 3rd Earl of Bute, lived from 1713 to 1792.
  Fact 0.19: John Stuart was Prime Minister from 26 May 1762 to 8 April 1763, for 318 days.
  Fact 0.20: John Stuart was a Tory.
  Fact 0.21: George Grenville was MP for Buckingham and lived from 1712 to 1770.
  Fact 0.22: George Grenville was Prime Minister from 16 April 1763 to 13 July 1765, for 2 years and 89 days.
  Fact 0.23: George Grenville was a Whig, in the Grenvillite faction.
  Fact 0.24: Charles Watson-Wentworth, 2nd Marquess of Rockingham, lived from 1730 to 1782.
  Fact 0.25: Charles Watson-Wentworth was Prime Minister from 13 July 1765 to 30 July 1766, for 1 year and 18 days.
  Fact 0.26: Charles Watson-Wentworth was a Whig, in the Rockinghamite faction.
  Fact 0.27: William Pitt, 1st Earl of Chatham, was MP for Bath and lived from 1708 to 1778.
  Fact 0.28: William Pitt, 1st Earl of Chatham, was Prime Minister from 30 July 1766 to 14 October 1768, for 2 years and 77 days.
  Fact 0.29: William Pitt, 1st Earl of Chatham, was a Whig, in the Chathamite faction.
  Fact 0.30: Augustus FitzRoy, 3rd Duke of Grafton, lived from 1735 to 1811.
  Fact 0.31: Augustus FitzRoy was Prime Minister from 14 October 1768 to 28 January 1770, for 1 year and 107 days.
  Fact 0.32: Augustus FitzRoy was a Whig, in the Chathamite faction.
  Fact 0.33: Frederick North, Lord North, was MP for Banbury and lived from 1732 to 1792.
  Fact 0.34: Frederick North was Prime Minister from 28 January 1770 to 22 March 1782, for 12 years and 54 days.
  Fact 0.35: Frederick North was a Tory, in the Northite faction.
  Fact 0.36: Charles Watson-Wentworth was Prime Minister again from 27 March 1782 to 1 July 1782, for 97 days.
  Fact 0.37: Charles Watson-Wentworth was a Whig in his second term, in the Rockinghamite faction.
  Fact 0.38: William Petty, 2nd Earl of Shelburne, lived from 1737 to 1805.
  Fact 0.39: William Petty was Prime Minister from 4 July 1782 to 2 April 1783, for 273 days.
  Fact 0.40: William Petty was a Whig, in the Chathamite faction.
  Fact 0.41: William Cavendish-Bentinck, 3rd Duke of Portland, lived from 1738 to 1809.
  Fact 0.42: William Cavendish-Bentinck was Prime Minister from 2 April 1783 to 19 December 1783, for 262 days.
  Fact 0.43: William Cavendish-Bentinck was a Whig.
  Fact 0.44: William Pitt the Younger was MP for Appleby and Cambridge University, and lived from 1759 to 1806.
  Fact 0.45: William Pitt the Younger was Prime Minister from 19 December 1783 to 14 March 1801, for 17 years and 86 days.
  Fact 0.46: William Pitt the Younger was a Tory, in the Pittite faction.
  Fact 0.47: Henry Addington was MP for Devizes and lived from 1757 to 1844.
  Fact 0.48: Henry Addington was Prime Minister from 17 March 1801 to 10 May 1804, for 3 years and 55 days.
  Fact 0.49: Henry Addington was a Tory, in the Pittite faction.
  Fact 0.50: William Pitt the Younger was Prime Minister again from 10 May 1804 to 23 January 1806, for 1 year and 259 days.
  Fact 0.51: William Pitt the Younger was a Tory, in the Pittite faction, in his second term.
  Fact 0.52: William Grenville, 1st Baron Grenville, lived from 1759 to 1834.
  Fact 0.53: William Grenville was Prime Minister from 11 February 1806 to 31 March 1807, for 1 year and 49 days.
  Fact 0.54: William Grenville was a Whig, in the All the Talents ministry.
  Fact 0.55: William Cavendish-Bentinck was Prime Minister again from 31 March 1807 to 4 October 1809, for 2 years and 188 days.
  Fact 0.56: William Cavendish-Bentinck was a Tory, in the Pittite faction, in his second term.
  Fact 0.57: Spencer Perceval was MP for Northampton and lived from 1762 to 1812.
  Fact 0.58: Spencer Perceval was Prime Minister from 4 October 1809 to 11 May 1812, for 2 years and 221 days.
  Fact 0.59: Spencer Perceval was a Tory, in the Pittite faction.
  Fact 0.60: Robert Jenkinson, 2nd Earl of Liverpool, lived from 1770 to 1828.
  Fact 0.61: Robert Jenkinson was Prime Minister from 8 June 1812 to 9 April 1827, for 14 years and 306 days.
  Fact 0.62: Robert Jenkinson was a Tory, in the Pittite faction.
  Fact 0.63: George Canning was MP for Seaford and lived from 1770 to 1827.
  Fact 0.64: George Canning was Prime Minister from 10 April 1827 to 8 August 1827, for 121 days.
  Fact 0.65: George Canning was a Tory, in the Canningite faction.
  Fact 0.66: Frederick John Robinson, 1st Viscount Goderich, lived from 1782 to 1859.
  Fact 0.67: Frederick John Robinson was Prime Minister from 31 August 1827 to 21 January 1828, for 144 days.
  Fact 0.68: Frederick John Robinson was a Tory, in the Canningite faction.
  Fact 0.69: Arthur Wellesley, 1st Duke of Wellington, lived from 1769 to 1852.
  Fact 0.70: Arthur Wellesley was Prime Minister from 22 January 1828 to 16 November 1830, for 2 years and 299 days.
  Fact 0.71: Arthur Wellesley was a Tory.
  Fact 0.72: Charles Grey, 2nd Earl Grey, lived from 1764 to 1845.
  Fact 0.73: Charles Grey was Prime Minister from 22 November 1830 to 9 July 1834, for 3 years and 230 days.
  Fact 0.74: Charles Grey was a Whig.
  Fact 0.75: William Lamb, 2nd Viscount Melbourne, lived from 1779 to 1848.
  Fact 0.76: William Lamb was Prime Minister from 16 July 1834 to 14 November 1834, for 122 days.
  Fact 0.77: William Lamb was a Whig.
  Fact 0.78: Arthur Wellesley was Prime Minister again from 14 November 1834 to 10 December 1834, for 27 days.
  Fact 0.79: Arthur Wellesley's second term was a caretaker ministry, and he was a Tory.
  Fact 0.80: Robert Peel was MP for Tamworth and lived from 1788 to 1850.
  Fact 0.81: Robert Peel was Prime Minister from 10 December 1834 to 8 April 1835, for 120 days.
  Fact 0.82: Robert Peel was a Conservative.
  Fact 0.83: William Lamb was Prime Minister again from 18 April 1835 to 30 August 1841, for 6 years and 135 days.
  Fact 0.84: William Lamb was a Whig in his second term.
  Fact 0.85: Robert Peel was Prime Minister again from 30 August 1841 to 29 June 1846, for 4 years and 304 days.
  Fact 0.86: Robert Peel was a Conservative in his second term.
  Fact 0.87: Lord John Russell was MP for the City of London and lived from 1792 to 1878.
  Fact 0.88: Lord John Russell was Prime Minister from 30 June 1846 to 21 February 1852, for 5 years and 237 days.
  Fact 0.89: Lord John Russell was a Whig.
  Fact 0.90: Edward Smith-Stanley, 14th Earl of Derby, lived from 1799 to 1869.
  Fact 0.91: Edward Smith-Stanley was Prime Minister from 23 February 1852 to 17 December 1852, for 299 days.
  Fact 0.92: Edward Smith-Stanley was a Conservative.
  Fact 0.93: George Hamilton-Gordon, 4th Earl of Aberdeen, lived from 1784 to 1860.
  Fact 0.94: George Hamilton-Gordon was Prime Minister from 19 December 1852 to 30 January 1855, for 2 years and 43 days.
  Fact 0.95: George Hamilton-Gordon was a Peelite.
  Fact 0.96: Henry John Temple, 3rd Viscount Palmerston, was MP for Tiverton and lived from 1784 to 1865.
  Fact 0.97: Henry John Temple was Prime Minister from 6 February 1855 to 19 February 1858, for 3 years and 14 days.
  Fact 0.98: Henry John Temple was a Whig.
  Fact 0.99: Edward Smith-Stanley was Prime Minister again from 20 February 1858 to 11 June 1859, for 1 year and 112 days.
  Fact 0.100: Edward Smith-Stanley was a Conservative in his second term.
  Fact 0.101: Henry John Temple was Prime Minister again from 12 June 1859 to 18 October 1865, for 6 years and 129 days.
  Fact 0.102: Henry John Temple was a Liberal in his second term.
  Fact 0.103: John Russell, 1st Earl Russell, lived from 1792 to 1878.
  Fact 0.104: John Russell, 1st Earl Russell, was Prime Minister from 29 October 1865 to 26 June 1866, for 241 days.
  Fact 0.105: John Russell, 1st Earl Russell, was a Liberal.
  Fact 0.106: Edward Smith-Stanley was Prime Minister again from 28 June 1866 to 25 February 1868, for 1 year and 243 days.
  Fact 0.107: Edward Smith-Stanley was a Conservative in his third term.
  Fact 0.108: Benjamin Disraeli was MP for Buckinghamshire and lived from 1804 to 1881.
  Fact 0.109: Benjamin Disraeli was Prime Minister from 27 February 1868 to 1 December 1868, for 279 days.
  Fact 0.110: Benjamin Disraeli was a Conservative.
  Fact 0.111: William Ewart Gladstone was MP for Greenwich and lived from 1809 to 1898.
  Fact 0.112: William Ewart Gladstone was Prime Minister from 3 December 1868 to 17 February 1874, for 5 years and 77 days.
  Fact 0.113: William Ewart Gladstone was a Liberal.
  Fact 0.114: Benjamin Disraeli, 1st Earl of Beaconsfield, was Prime Minister again from 20 February 1874 to 21 April 1880, for 6 years and 62 days.
  Fact 0.115: Benjamin Disraeli was a Conservative in his second term.
  Fact 0.116: William Ewart Gladstone was MP for Midlothian and was Prime Minister again from 23 April 1880 to 9 June 1885, for 5 years and 48 days.
  Fact 0.117: William Ewart Gladstone was a Liberal in his second term.
  Fact 0.118: Robert Gascoyne-Cecil, 3rd Marquess of Salisbury, lived from 1830 to 1903.
  Fact 0.119: Robert Gascoyne-Cecil was Prime Minister from 23 June 1885 to 28 January 1886, for 220 days.
  Fact 0.120: Robert Gascoyne-Cecil was a Conservative.
  Fact 0.121: William Ewart Gladstone was Prime Minister again from 1 February 1886 to 20 July 1886, for 170 days.
  Fact 0.122: William Ewart Gladstone was a Liberal in his third term.
  Fact 0.123: Robert Gascoyne-Cecil was Prime Minister again from 25 July 1886 to 11 August 1892, for 6 years and 18 days.
  Fact 0.124: Robert Gascoyne-Cecil was a Conservative in his second term.
  Fact 0.125: William Ewart Gladstone was Prime Minister again from 15 August 1892 to 2 March 1894, for 1 year and 200 days.
  Fact 0.126: William Ewart Gladstone was a Liberal in his fourth term.
  Fact 0.127: Archibald Primrose, 5th Earl of Rosebery, lived from 1847 to 1929.
  Fact 0.128: Archibald Primrose was Prime Minister from 5 March 1894 to 22 June 1895, for 1 year and 110 days.
  Fact 0.129: Archibald Primrose was a Liberal.
  Fact 0.130: Robert Gascoyne-Cecil was Prime Minister again from 25 June 1895 to 11 July 1902, for 7 years and 17 days.
  Fact 0.131: Robert Gascoyne-Cecil's third term was labelled Con. and Lib. U. in the table.
  Fact 0.132: Arthur Balfour was MP for Manchester East and lived from 1848 to 1930.
  Fact 0.133: Arthur Balfour was Prime Minister from 11 July 1902 to 5 December 1905, for 3 years and 148 days.
  Fact 0.134: Arthur Balfour was a Conservative.
  Fact 0.135: Henry Campbell-Bannerman was MP for Stirling Burghs and lived from 1836 to 1908.
  Fact 0.136: Henry Campbell-Bannerman was Prime Minister from 5 December 1905 to 7 April 1908, for 2 years and 125 days.
  Fact 0.137: Henry Campbell-Bannerman was a Liberal.
  Fact 0.138: Herbert Henry Asquith was MP for East Fife and lived from 1852 to 1928.
  Fact 0.139: Herbert Henry Asquith was Prime Minister from 7 April 1908 to 5 December 1916, for 8 years and 243 days.
  Fact 0.140: Herbert Henry Asquith was a Liberal.
  Fact 0.141: David Lloyd George was MP for Caernarvon Boroughs and lived from 1863 to 1945.
  Fact 0.142: David Lloyd George was Prime Minister from 6 December 1916 to 19 October 1922, for 5 years and 318 days.
  Fact 0.143: David Lloyd George was a Liberal.
  Fact 0.144: Bonar Law was MP for Glasgow Central and lived from 1858 to 1923.
  Fact 0.145: Bonar Law was Prime Minister from 23 October 1922 to 20 May 1923, for 210 days.
  Fact 0.146: Bonar Law was a Conservative.
  Fact 0.147: Stanley Baldwin was MP for Bewdley and lived from 1867 to 1947.
  Fact 0.148: Stanley Baldwin was Prime Minister from 23 May 1923 to 16 January 1924, for 239 days.
  Fact 0.149: Stanley Baldwin was a Conservative.
  Fact 0.150: Ramsay MacDonald was MP for Aberavon and lived from 1866 to 1937.
  Fact 0.151: Ramsay MacDonald was Prime Minister from 22 January 1924 to 4 November 1924, for 288 days.
  Fact 0.152: Ramsay MacDonald was a Labour politician.
  Fact 0.153: Stanley Baldwin was Prime Minister again from 4 November 1924 to 5 June 1929, for 4 years and 214 days.
  Fact 0.154: Stanley Baldwin was a Conservative in his second term.
  Fact 0.155: Ramsay MacDonald was MP for Seaham and was Prime Minister again from 5 June 1929 to 7 June 1935, for 6 years and 3 days.
  Fact 0.156: Ramsay MacDonald was a Labour politician in his second term.
  Fact 0.157: Stanley Baldwin was Prime Minister again from 7 June 1935 to 28 May 1937, for 1 year and 356 days.
  Fact 0.158: Stanley Baldwin was a Conservative in his third term.
  Fact 0.159: Neville Chamberlain was MP for Birmingham Edgbaston and lived from 1869 to 1940.
  Fact 0.160: Neville Chamberlain was Prime Minister from 28 May 1937 to 10 May 1940, for 2 years and 349 days.
  Fact 0.161: Neville Chamberlain was a Conservative.
  Fact 0.162: Winston Churchill was MP for Epping and lived from 1874 to 1965.
  Fact 0.163: Winston Churchill was Prime Minister from 10 May 1940 to 26 July 1945, for 5 years and 78 days.
  Fact 0.164: Winston Churchill was a Conservative.
  Fact 0.165: Clement Attlee was MP for Limehouse and Walthamstow West and lived from 1883 to 1967.
  Fact 0.166: Clement Attlee was Prime Minister from 26 July 1945 to 26 October 1951, for 6 years and 93 days.
  Fact 0.167: Clement Attlee was a Labour politician.
  Fact 0.168: Winston Churchill was MP for Woodford and was Prime Minister again from 26 October 1951 to 6 April 1955, for 3 years and 163 days.
  Fact 0.169: Winston Churchill was a Conservative in his second term.
  Fact 0.170: Anthony Eden was MP for Warwick and Leamington and lived from 1897 to 1977.
  Fact 0.171: Anthony Eden was Prime Minister from 6 April 1955 to 10 January 1957, for 1 year and 280 days.
  Fact 0.172: Anthony Eden was a Conservative.
  Fact 0.173: Harold Macmillan was MP for Bromley and lived from 1894 to 1986.
  Fact 0.174: Harold Macmillan was Prime Minister from 10 January 1957 to 19 October 1963, for 6 years and 283 days.
  Fact 0.175: Harold Macmillan was a Conservative.
  Fact 0.176: Alec Douglas-Home was MP for Kinross and Western Perthshire and lived from 1903 to 1995.
  Fact 0.177: Alec Douglas-Home was Prime Minister from 19 October 1963 to 16 October 1964, for 364 days.
  Fact 0.178: Alec Douglas-Home was a Conservative, in the Scottish Unionist faction.
  Fact 0.179: Harold Wilson was MP for Huyton and lived from 1916 to 1995.
  Fact 0.180: Harold Wilson was Prime Minister from 16 October 1964 to 19 June 1970, for 5 years and 247 days.
  Fact 0.181: Harold Wilson was a Labour politician.
  Fact 0.182: Edward Heath was MP for Bexley and Sidcup and lived from 1916 to 2005.
  Fact 0.183: Edward Heath was Prime Minister from 19 June 1970 to 4 March 1970, as the table lists it, for 3 years and 259 days.
  Fact 0.184: Edward Heath was a Conservative.
  Fact 0.185: Harold Wilson was Prime Minister again from 4 March to 5 April 1976, for 2 years and 33 days.
  Fact 0.186: Harold Wilson was a Labour politician in his second term.
  Fact 0.187: James Callaghan was MP for Cardiff South East and lived from 1912 to 2005.
  Fact 0.188: James Callaghan was Prime Minister from 5 April 1976 to 4 May 1979, for 3 years and 30 days.
  Fact 0.189: James Callaghan was a Labour politician.
  Fact 0.190: Margaret Thatcher was MP for Finchley and lived from 1925 to 2013.
  Fact 0.191: Margaret Thatcher was Prime Minister from 4 May 1979 to 28 November 1990, for 11 years and 209 days.
  Fact 0.192: Margaret Thatcher was a Conservative.
  Fact 0.193: John Major was MP for Huntingdon and was born in 1943.
  Fact 0.194: John Major was Prime Minister from 28 November 1990 to 2 May 1997, for 6 years and 156 days.
  Fact 0.195: John Major was a Conservative.
  Fact 0.196: Tony Blair was MP for Sedgefield and was born in 1953.
  Fact 0.197: Tony Blair was Prime Minister from 2 May 1997 to 27 June 2007, for 10 years and 57 days.
  Fact 0.198: Tony Blair was a Labour politician.
  Fact 0.199: Gordon Brown was MP for Kirkcaldy and Cowdenbeath and was born in 1951.
  Fact 0.200: Gordon Brown was Prime Minister from 27 June 2007 to 11 May (the year is not shown in the table), for 2 years and 319 days.
  Fact 0.201: Gordon Brown was a Labour politician.
  Fact 0.202: David Cameron was MP for Witney and was born in 1966.
  Fact 0.203: David Cameron was Prime Minister from 11 May (the year is not shown in the table) to 13 July 2016, for 6 years and 64 days.
  Fact 0.204: David Cameron was a Conservative, in a coalition with the Liberal Democrats.
  Fact 0.205: Theresa May was MP for Maidenhead and was born in 1956.
  Fact 0.206: Theresa May has been Prime Minister since 13 July 2016.
  Fact 0.207: Theresa May's term was 1 year and 193 days as of the table.
  Fact 0.208: Theresa May was a Conservative.
  Fact 0.209: The monarch Elizabeth II reigned from 1952 to the present.
  Fact 0.210: The monarch George I reigned from 1714 to 1727.
  Fact 0.211: The monarch George II reigned from 1727 to 1760.
  Fact 0.212: The monarch George III reigned from 1760 to 1820.
  Fact 0.213: The monarch George IV reigned from 1820 to 1830.
  Fact 0.214: The monarch William IV reigned from 1830 to 1837.
  Fact 0.215: The monarch Victoria reigned from 1837 to 1901.
  Fact 0.216: The monarch Edward VII reigned from 1901 to 1910.
  Fact 0.217: The monarch George V reigned from 1910 to 1936.
  Fact 0.218: The monarch Edward VIII reigned in 1936.
  Fact 0.219: The monarch George VI reigned from 1936 to 1952.

Retrieved items:
<item 1>
List of Prime Ministers of the United Kingdom
V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name (
Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert
Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the
Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741
7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743
-- First Lord of the Treasury Whig Carteret
</item 1>

<item 2>
List of Prime Ministers of the United Kingdom
^ 7 Majority Conservative government before dissolution . See also ( edit ) United Kingdom This article is part of a series on the politics and government of
the United Kingdom Constitution ( show ) Constitution Cabinet Manual Erskine May Taxation The Crown ( show ) British Monarchy Monarch Queen Elizabeth II Heir
apparent The Prince of Wales Royal family Succession to the British throne Royal prerogative Executive ( show ) HM Government Prime Minister Theresa May Cabinet
Current ministry Her Majesty 's Civil Service Departments Executive agencies Public bodies Privy Council Crown - in - Council Queen Elizabeth II Legislature (
show ) Parliament ( 56th Parliament ) Crown - in - Parliament Queen Elizabeth II House of Lords Lord Speaker The Lord Fowler House of Commons Speaker John
Bercow Leader of the Opposition Jeremy Corbyn Her Majesty 's Loyal Opposition Shadow Cabinet State Opening of Parliament Prime Minister 's Questions Act of
Parliament Statutory Instruments Judiciary ( show ) Supreme Court President The Baroness Hale of Richmond Deputy President The Lord Mance Judges of the Supreme
Court Courts of the United Kingdom Courts of England and Wales Courts of Scotland Courts of Northern Ireland Devolution ( show ) Joint Ministerial Committee
Scotland Government Parliament Wales Government National Assembly Reserved matters Northern Ireland Executive Assembly Administrative geography ( show )
</item 2>

<item 3>
List of Prime Ministers of the United Kingdom
List of Prime Ministers of the United Kingdom - wikipedia List of Prime Ministers of the United Kingdom Jump to : navigation , search Top left : Walpole became
the first Prime Minister of the Kingdom of Great Britain in 1721 . Top right : Pitt the Younger became the first Prime Minister of the United Kingdom of Great
Britain and Ireland in 1801 . Bottom left : Lloyd George was the last Prime Minister of the United Kingdom of Great Britain and Ireland as a result of the Anglo
- Irish Treaty in 1922 . Bottom right : Baldwin was Prime Minister when the country was renamed the United Kingdom of Great Britain and Northern Ireland in 1927
. The Prime Minister of the United Kingdom is the head of the Government of the United Kingdom , and chairs Cabinet meetings . There is no specific date when
the office of Prime Minister first appeared , as the role was not created but rather evolved over a period of time . The term was used in the House of Commons
in 1805 and it was certainly in parliamentary use by the 1880s , and in 1905 the post of Prime Minister was officially given recognition in the order of
precedence .
</item 3>

<item 4>
List of Prime Ministers of the United Kingdom
-- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 --
7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the
Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and
77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord
of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April
1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William
</item 4>

<item 5>
List of Prime Ministers of the United Kingdom
14 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 17 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 28 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 34 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 36 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 42 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 46 . Jump up
^ Eccleshall & Walker ( 2002 ) , p. 54 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 61 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 64 . Jump up ^ Eccleshall
& Walker ( 2002 ) , p. 73 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 77 .
</item 5>

<item 6>
List of Prime Ministers of the United Kingdom
77 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 85 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 94 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 98 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 101 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 106 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 116 . Jump up ^
Eccleshall & Walker ( 2002 ) , p. 120 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 123 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 128 . ^ Jump up to :
Eccleshall & Walker ( 2002 ) , p. 136 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 142 .
</item 6>

<item 7>
List of Prime Ministers of the United Kingdom
BBC News . 13 July 2016 . Retrieved 26 July 2016 . Lay summary ( 13 July 2016 ) . Bibliography ( edit ) Burt , Llewellyn C. ( 1874 ) . A synoptical history of
England ( 2nd ed . ) . London : Lockwood . Carpenter , Clive , ed. ( 1992 ) . The Guinness UK Data Book . Guinness . ISBN 978 - 0 - 85112 - 522 - 0 . Chapman ,
Richard A. ( 31 January 2002 ) . `` History : from earliest times to the present day '' . The Treasury in Public Policy - Making . Routledge . ISBN 978 - 1 -
134 - 86426 - 3 . Clarke , John ( 1999 ) . `` House of Hanover '' . In Fraser , Antonia . The Lives of the Kings & Queens of England . University of California
Press ( published February 2000 ) .
</item 7>

<item 8>
List of Prime Ministers of the United Kingdom
After the succession of King George I in 1714 , the arrangement of a commission of Lords of the Treasury as opposed to a single Lord High Treasurer became
permanent . For the next three years , the government was headed by Lord Townshend , who was appointed Secretary of State for the Northern Department .
Subsequently , Lord Stanhope and Lord Sunderland ran the government jointly , with Stanhope managing foreign affairs and Sunderland managing domestic affairs .
Stanhope died in February 1721 and Sunderland resigned two months later ; Townshend and Robert Walpole were then invited to form the next government . From that
point , the holder of the office of First Lord also unofficially held the status of Prime Minister . It was not until the Edwardian era that the title Prime
Minister was constitutionally recognised . With only few exceptions , the Prime Minister still holds the office of First Lord by constitutional convention .
Since 1721 ( edit ) By century and monarch 18th to 19th century George I George II George III George IV William IV Victoria 20th to 21st century Edward VII
George V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative (
</item 8>

<item 9>
List of Prime Ministers of the United Kingdom
2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the
Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable
Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July
2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days Addenda ( edit ) ^ 1 Whig majority in the Commons before the dissolution of Parliament . ^
2 Minority government before dissolution . ^ 3 Tory majority in the Commons before dissolution . ^ 4 Majority Tory government before dissolution . ^ 5 Majority
Liberal government before dissolution . ^ 6 Coalition government during the course of an undissolved hung parliament . ^ 7 Majority Conservative government
before dissolution . See
</item 9>

<item 10>
List of Prime Ministers of the United Kingdom
-- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS
MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​
Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26
July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker (
Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26
July
</item 10>
=== END ITEM id=97242bf8a40f049f ===

=== ITEM id=f4b1e66a3633952b ===
Gold sentences and facts:
Sentence 0: The total number of languages natively spoken in Africa is variously estimated ( depending on the delineation of language vs. dialect ) at between 1,250 to 2,100 , and by some counts at `` over 3,000 '' , Nigeria alone has over 500 languages ( according to the count of SIL Ethnologue ) , one of the greatest concentrations of linguistic diversity in the world .
  Fact 0.0: The total number of languages natively spoken in Africa is variously estimated at between 1,250 and 2,100, depending on the delineation of language vs. dialect.
  Fact 0.1: Some counts put the number of languages natively spoken in Africa at 'over 3,000'.
  Fact 0.2: Nigeria alone has over 500 languages, according to the count of SIL Ethnologue.
  Fact 0.3: Nigeria is one of the greatest concentrations of linguistic diversity in the world.

Retrieved items:
<item 1>
Languages of Africa
Case , tense and other categories may be distinguished only by tone . Semantic ( edit ) Quite often , only one term is used for both animal and meat ; the word
nama or nyama for animal / meat is particularly widespread in otherwise widely divergent African languages . Number of speakers ( edit ) The following is a
table displaying the number of speakers of given languages within Africa : Language Family Native speakers ( L1 ) Official status per country Afrikaans Indo -
European 7,200,000 South Africa Akan Niger -- Congo 11,000,000 None . Government sponsored language of Ghana Amharic Afroasiatic 21,800,000 Ethiopia Arabic
Afroasiatic 150,000,000 but with separate mutually unintelligible varieties Algeria , Chad , Comoros , Djibouti , Egypt , Eritrea , Libya , Mauritania , Morocco
, Somalia , Sudan , Tunisia Berber Afroasiatic 56,000,000 ( estimated ) ( including separate unintelligible varieties ) Morocco , Algeria Chewa Niger -- Congo
9,700,000 Malawi , Zimbabwe English Indo - European 6,500,
</item 1>

<item 2>
Sub-Saharan Africa
Life ( Exp . ) ( Life Expectancy 2006 ) , Literacy ( Male / Female 2006 ) , Trans ( Transparency 2009 ) , HDI ( Human Development Index ) , EODBR ( Ease of
Doing Business Rank June 2008 through May 2009 ) , SAB ( Starting a Business June 2008 through May 2009 ) , PFI ( Press Freedom Index 2009 ) Languages and
ethnic groups ( edit ) Further information : Languages of Africa and List of African ethnic groups Map showing the traditional language families spoken in
Africa : Afroasiatic Austronesian Indo - European Khoisan Niger - Congo Nilo - Saharan A San man ( Khoisan ) An Afrikaner family ( Indo - European ) Saho women
( Afroasiatic ) Maasai women and children ( Nilo - Saharan ) Yoruba drummers ( Niger - Congo ) Sub-Saharan Africa contains over 1,000 languages , which is
around 1 / 6 of the world 's total .
</item 2>

<item 3>
Languages of Africa
135,000 native language of Mauritius Mossi Niger -- Congo 7,600,000 Recognised regional language in Burkina Faso Ndebele Niger -- Congo 1,090,000 Statuatory
national language in South Africa Northern Sotho Niger -- Congo 4,600,000 South Africa Oromo Afroasiatic 26,000,000 Ethiopia Portuguese Indo - European
13,700,000 ( estimated ) Angola , Cape Verde , Guinea - Bissau , Equatorial Guinea , Mozambique , São Tomé and Príncipe Sesotho Niger -- Congo 5,600,000 Lesotho
, South Africa , Zimbabwe Shona Niger -- Congo 14,200,000 incl . Manyika , Ndau ( 2000 -- 2006 ) Zimbabwe Somali Afroasiatic 16,600,000 Somalia Spanish Indo -
European 2,700,000 Equatorial Guinea Swahili Niger -- Congo 15,000,000 official in Tanzania , Kenya , Uganda , Rwanda national language of Democratic Republic
of Congo Tigrinya Afroasiatic 7,000,000 Eritrea Tshiluba Niger -- Congo 6,300,
</item 3>

<item 4>
Languages of Africa
000,000 Eritrea Tshiluba Niger -- Congo 6,300,000 ( 1991 ) national language of Democratic Republic of the Congo Tswana Niger -- Congo 5,800,000 South Africa ,
Botswana Umbundu Niger -- Congo 6,000,000 recognised national language of Angola Xhosa Niger -- Congo 7,600,000 South Africa , Zimbabwe Yoruba Niger -- Congo
28,000,000 Zulu Niger -- Congo 10,400,000 South Africa See also ( edit ) Africa portal Language portal General ( edit ) Languages of the African Union Writing
systems of Africa Journal of West African Languages Works ( edit ) Polyglotta Africana The Languages of Africa Classifiers ( edit ) Karl Lepsius Wilhelm Bleek
Carl Meinhof Diedrich Westermann Joseph Greenberg Notes ( edit ) Jump up ^ Heine , Bernd ; Heine , Bernd , eds. ( 2000 ) . African Languages : an Introduction .
Cambridge University Press . Jump up ^ Epstein , Edmund L. ; Kole , Robert , eds. ( 1998 ) . The Language of African Literature . Africa World Press .
</item 4>

<item 5>
Languages of Africa
5 Unclassified languages 1.4. 6 Sign languages 2 Language in Africa 2.1 Official Languages 2.2 Cross-border languages 2.3 Language change and planning 2.4
Demographics 3 Linguistic features 3.1 Phonological 3.2 Syntactic 3.3 Semantic 4 Number of speakers 5 See also 5.1 General 5.2 Works 5.3 Classifiers 6 Notes 7
References 8 External links Language groups ( edit ) Clickable map showing the traditional language families , subfamilies and major languages spoken in Africa
Most languages spoken in Africa belong to one of three large language families : Afroasiatic , Nilo - Saharan and Niger -- Congo . Another hundred belong to
small families such as Ubangian ( sometimes grouped within Niger - Congo ) and the various families called Khoisan , or the Indo - European and Austronesian
language families mainly spoken outside Africa ; the presence of the latter two dates to 2,600 and 1,500 years ago , respectively . In addition , the languages
of Africa languages include several unclassified languages and sign languages .
</item 5>

<item 6>
Languages of Africa
English , French and Portuguese are important languages in Africa . About 130 million , 115 million and 30 million Africans , respectively , speak them as
either native or secondary languages . Portuguese has become the national language of Angola . Through ( among other factors ) sheer demographic weight ,
Africans are increasingly taking ownership of these three world languages and having an ever greater influence on their development and growth . Linguistic
features ( edit ) Some linguistic features are particularly common among languages spoken in Africa , whereas others are less common . Such shared traits
probably are not due to a common origin of all African languages . Instead , some may be due to language contact ( resulting in borrowing ) and specific idioms
and phrases may be due to a similar cultural background . Phonological ( edit ) Some widespread phonetic features include : certain types of consonants , such
as implosives ( / ɓa / ) , ejectives ( / kʼa / ) , the labiodental flap and in southern Africa , clicks ( / ǂa / , / ŋǃa / ) . True implosives are rare outside
Africa , and clicks and the flap almost unheard of .
</item 6>

<item 7>
Languages of Africa
Languages of Africa - wikipedia Languages of Africa Jump to : navigation , search The traditional language families spoken in Africa : Afroasiatic Nilo -
Saharan Niger -- Congo Bantu Khoisan Indo - European Austronesian The languages of Africa are divided into six major language families : Afroasiatic languages
are spread throughout Western Asia , North Africa , the Horn of Africa and parts of the Sahel . Austronesian languages are spoken in Madagascar . Indo -
European languages are spoken in South Africa and Namibia ( Afrikaans , English , German ) and are used as lingua francas in the former colonies of Britain (
English ) , former colonies of France ( French ) , former colonies of Portugal ( Portuguese ) , former colonies of Spain ( Spanish ) and the current Spanish
territories of Ceuta , Melilla and the Canary Islands ( Spanish ) . Khoe languages are concentrated in the Kalahari Desert of Namibia and Botswana . Niger --
Congo languages ( Bantu and non-Bantu ) cover West , Central , Southeast and Southern Africa . Nilo - Saharan languages ( unity debated ) are spoken from
Tanzania to Sudan and from Chad to Mali .
</item 7>

<item 8>
Languages of Africa
There are several other small families and language isolates , as well as languages that have yet to be classified . In addition , Africa has a wide variety of
sign languages , many of which are language isolates ( see below ) . The total number of languages natively spoken in Africa is variously estimated ( depending
on the delineation of language vs. dialect ) at between 1,250 to 2,100 , and by some counts at `` over 3,000 '' , Nigeria alone has over 500 languages (
according to the count of SIL Ethnologue ) , one of the greatest concentrations of linguistic diversity in the world . Around a hundred languages are widely
used for inter-ethnic communication . Arabic , Somali , Berber , Amharic , Oromo , Igbo , Swahili , Hausa , Manding , Fulani and Yoruba are spoken by tens of
millions of people . Twelve dialect clusters ( which may group up to a hundred linguistic varieties ) are spoken by 75 percent , and fifteen by 85 percent , of
Africans as a first or additional language .
</item 8>

<item 9>
Languages of Africa
Trade languages are another age - old phenomenon in the African linguistic landscape . Cultural and linguistic innovations spread along trade routes and
languages of peoples dominant in trade developed into languages of wider communication ( lingua franca ) . Of particular importance in this respect are Berber (
North and West Africa ) , Jula ( western West Africa ) , Fulfulde ( West Africa ) , Hausa ( West Africa ) , Lingala ( Congo ) , Swahili ( Southeast Africa ) ,
Somali ( Horn of Africa ) and Arabic ( North Africa and Horn of Africa ) . After gaining independence , many African countries , in the search for national
unity , selected one language , generally the former colonial language , to be used in government and education . However , in recent years , African countries
have become increasingly supportive of maintaining linguistic diversity . Language policies that are being developed nowadays are mostly aimed at
multilingualism . Official languages ( edit ) Official languages in Africa : Afrikaans Portuguese Arabic Spanish English Swahili French other languages Besides
the former colonial languages of English , French , Portuguese and Spanish , the following languages are official at the national level in Africa (
non-exhaustive list ) : Afroasiatic Arabic in Comoros , Chad ,
</item 9>

<item 10>
Languages of Africa
In addition to the ( likely modest ) impact of borders , there are also cases of dialect levelling ( such as in Igbo and probably many others ) , koinés ( such
as N'Ko and possibly Runyakitara ) and emergence of new dialects ( such as Sheng ) . In some countries , there are official efforts to develop standardized
language versions . There are also many less widely spoken languages that may be considered endangered languages . Demographics ( edit ) Further information :
Demographics of Africa Of the 1 billion Africans ( in 2009 ) , about 17 percent speak an Arabic dialect . About 10 percent speak Swahili , the lingua franca of
Southeast Africa ; about 5 percent speak a Berber dialect ; and about 5 percent speak Hausa , which serves as a lingua franca in much of the Sahel . Other
important West African languages are Yoruba , Igbo and Fula . Major Horn of Africa languages are Amharic , Oromo and Somali . Important South African languages
are Zulu , Xhosa and Afrikaans . English , French and Portuguese are important languages in Africa .
</item 10>
=== END ITEM id=f4b1e66a3633952b ===

=== ITEM id=e6abad7fc50a58ef ===
Gold sentences and facts:
Sentence 0: 2017 World Series Team ( Wins ) Manager ( s ) Season Houston Astros ( 4 ) A.J. Hinch 101 -- 61 (. 623 ) Los Angeles Dodgers ( 3 ) Dave Roberts 104 -- 58 (. 642 ) Dates October 24 -- November 1 MVP George Springer ( Houston ) Umpires Phil Cuzzi ( Games 1 -- 2 ) , Gerry Davis ( crew chief ) , Laz Díaz , Dan Iassogna , Bill Miller , Paul Nauert , Mark Wegner ( Games 3 -- 7 ) ALCS Houston Astros defeated New York Yankees , 4 -- 3 NLCS Los Angeles Dodgers defeated Chicago Cubs , 4 -- 1 Broadcast Television Fox Fox Deportes MLB International TV announcers Joe Buck , John Smoltz , Ken Rosenthal , and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez , and Jaime Motta ( Fox Deportes ) Matt Vasgersian and Buck Martinez ( MLB International ) Radio ESPN Radio ESPN Deportes Radio Sportstalk 790 ( HOU -- English ) FM 94.1 HD3 ( HOU -- Spanish ) AM 570 LA Sports ( LAD -- English ) Univision America 1020 ( LAD -- Spanish ) Radio Korea 1540 ( LAD -- Korean ) Radio announcers Dan Shulman and Aaron Boone ( ESPN ) Eduardo Ortega , Renato Bermúdez , José Francisco Rivera , and Orlando Hernández ( ESPN Deportes ) Robert Ford and Steve Sparks ( HOU -- English ) Alex Treviño and Francisco Romero ( HOU -- Spanish ) Charley Steiner and Rick Monday ( LAD -- English ) Jaime Jarrín and Jorge Jarrín ( LAD -- Spanish ) Richard Choi and Chong Ho Yim ( LAD -- Korean ) World Series Program ← 2016 World Series
  Fact 0.0: The 2017 World Series table lists the Houston Astros as the team with 4 wins.
  Fact 0.1: The 2017 World Series table lists the Los Angeles Dodgers as the team with 3 wins.
  Fact 0.2: A.J. Hinch managed the Houston Astros in the 2017 season, with a record of 101 wins and 61 losses (.623).
  Fact 0.3: Dave Roberts managed the Los Angeles Dodgers in the 2017 season, with a record of 104 wins and 58 losses (.642).
  Fact 0.4: The 2017 World Series was played from October 24 to November 1.
  Fact 0.5: George Springer of the Houston Astros was the MVP of the 2017 World Series.
  Fact 0.6: Phil Cuzzi was an umpire for Games 1 and 2 of the 2017 World Series.
  Fact 0.7: Gerry Davis was the crew chief umpire of the 2017 World Series.
  Fact 0.8: Laz Diaz was an umpire of the 2017 World Series.
  Fact 0.9: Dan Iassogna was an umpire of the 2017 World Series.
  Fact 0.10: Bill Miller was an umpire of the 2017 World Series.
  Fact 0.11: Paul Nauert was an umpire of the 2017 World Series.
  Fact 0.12: Mark Wegner was an umpire for Games 3 to 7 of the 2017 World Series.
  Fact 0.13: The Houston Astros defeated the New York Yankees 4 to 3 in the 2017 ALCS.
  Fact 0.14: The Los Angeles Dodgers defeated the Chicago Cubs 4 to 1 in the 2017 NLCS.
  Fact 0.15: The 2017 World Series was broadcast on Fox television.
  Fact 0.16: The 2017 World Series was broadcast on Fox Deportes.
  Fact 0.17: The 2017 World Series was broadcast on MLB International TV.
  Fact 0.18: Joe Buck, John Smoltz, Ken Rosenthal and Tom Verducci were TV announcers on Fox.
  Fact 0.19: Rolando Nichols, Carlos Alvarez, Edgar Gonzalez and Jaime Motta were TV announcers on Fox Deportes.
  Fact 0.20: Matt Vasgersian and Buck Martinez were TV announcers on MLB International.
  Fact 0.21: The 2017 World Series was broadcast on ESPN Radio.
  Fact 0.22: The 2017 World Series was broadcast on ESPN Deportes Radio.
  Fact 0.23: Sportstalk 790 carried the Houston Astros English radio broadcast.
  Fact 0.24: FM 94.1 HD3 carried the Houston Astros Spanish radio broadcast.
  Fact 0.25: AM 570 LA Sports carried the Los Angeles Dodgers English radio broadcast.
  Fact 0.26: Univision America 1020 carried the Los Angeles Dodgers Spanish radio broadcast.
  Fact 0.27: Radio Korea 1540 carried the Los Angeles Dodgers Korean radio broadcast.
  Fact 0.28: Dan Shulman and Aaron Boone were radio announcers on ESPN.
  Fact 0.29: Eduardo Ortega, Renato Bermudez, Jose Francisco Rivera and Orlando Hernandez were radio announcers on ESPN Deportes.
  Fact 0.30: Robert Ford and Steve Sparks were English radio announcers for the Houston Astros.
  Fact 0.31: Alex Trevino and Francisco Romero were Spanish radio announcers for the Houston Astros.
  Fact 0.32: Charley Steiner and Rick Monday were English radio announcers for the Los Angeles Dodgers.
  Fact 0.33: Jaime Jarrin and Jorge Jarrin were Spanish radio announcers for the Los Angeles Dodgers.
  Fact 0.34: Richard Choi and Chong Ho Yim were Korean radio announcers for the Los Angeles Dodgers.
  Fact 0.35: The 2017 World Series followed the 2016 World Series.
Sentence 1: The 2017 World Series was the championship series of Major League Baseball 's ( MLB ) 2017 season . The 113th edition of the World Series , it was played between October 24 and November 1 . The series was a best - of - seven playoff between the National League ( NL ) champion Los Angeles Dodgers and the American League ( AL ) champion Houston Astros . It was sponsored by the internet television service YouTube TV and officially known as the World Series presented by YouTube TV .
  Fact 1.0: The 2017 World Series was the championship series of Major League Baseball's 2017 season.
  Fact 1.1: The 2017 World Series was the 113th edition of the World Series.
  Fact 1.2: The 2017 World Series was played between October 24 and November 1.
  Fact 1.3: The 2017 World Series was a best-of-seven playoff.
  Fact 1.4: The 2017 World Series was played between the National League champion Los Angeles Dodgers.
  Fact 1.5: The 2017 World Series was played between the American League champion Houston Astros.
  Fact 1.6: The 2017 World Series was sponsored by the internet television service YouTube TV.
  Fact 1.7: The 2017 World Series was officially known as the World Series presented by YouTube TV.

Retrieved items:
<item 1>
2017 World Series
`` Overview of World Series home - field advantage '' . Major League Baseball . Retrieved November 2 , 2017 . We are just a week away from the World Series
presented by YouTube TV Jump up ^ `` Astros , Dodgers complete record - setting World Series '' . ESPN . November 2 , 2017 . Retrieved November 2 , 2017 . ^
Jump up to : Castrovince , Anthony ( November 1 , 2017 ) . `` Springer swats his way to MVP honors '' . Major League Baseball . Retrieved November 2 , 2017 .
Jump up ^ Schoenfield , David ( November 2 , 2017 ) . `` George Springer wins World Series MVP , ties record with five homers '' . ESPN . Retrieved November 2 ,
2017 . Jump up ^ Whicker , Mark ( October 22 , 2017 ) . `` Whicker : Dodgers , Houston Astros try to rekindle their 80s rivalry '' . Orange County Register .
Retrieved October 25 , 2017 .
</item 1>

<item 2>
The 2017 World Series (October 24–November 1) pitted the 104-win Los Angeles Dodgers against the 101-win Houston Astros, with home-field advantage determined by
regular-season record for the first time. Houston won 4–3, earning its first franchise championship and making history as Texas’s first World Series-winning
team. George Springer was MVP, tying the single-series record with five home runs. The teams combined for a record 25 homers, including eight in Game
</item 2>

<item 3>
2017 World Series
and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez , and Jaime Motta ( Fox Deportes ) Matt Vasgersian and Buck Martinez ( MLB
International ) Radio ESPN Radio ESPN Deportes Radio Sportstalk 790 ( HOU -- English ) FM 94.1 HD3 ( HOU -- Spanish ) AM 570 LA Sports ( LAD -- English )
Univision America 1020 ( LAD -- Spanish ) Radio Korea 1540 ( LAD -- Korean ) Radio announcers Dan Shulman and Aaron Boone ( ESPN ) Eduardo Ortega , Renato
Bermúdez , José Francisco Rivera , and Orlando Hernández ( ESPN Deportes ) Robert Ford and Steve Sparks ( HOU -- English ) Alex Treviño and Francisco Romero (
HOU -- Spanish ) Charley Steiner and Rick Monday ( LAD -- English ) Jaime Jarrín and Jorge Jarrín ( LAD -- Spanish ) Richard Choi and Chong Ho Yim ( LAD --
Korean ) World Series Program ← 2016 World Series The 2017 World Series was the championship series of Major League Baseball 's ( MLB ) 2017 season .
</item 3>

<item 4>
2017 World Series
2017 World Series - wikipedia 2017 World Series Jump to : navigation , search 2017 World Series Team ( Wins ) Manager ( s ) Season Houston Astros ( 4 ) A.J.
Hinch 101 -- 61 (. 623 ) Los Angeles Dodgers ( 3 ) Dave Roberts 104 -- 58 (. 642 ) Dates October 24 -- November 1 MVP George Springer ( Houston ) Umpires Phil
Cuzzi ( Games 1 -- 2 ) , Gerry Davis ( crew chief ) , Laz Díaz , Dan Iassogna , Bill Miller , Paul Nauert , Mark Wegner ( Games 3 -- 7 ) ALCS Houston Astros
defeated New York Yankees , 4 -- 3 NLCS Los Angeles Dodgers defeated Chicago Cubs , 4 -- 1 Broadcast Television Fox Fox Deportes MLB International TV announcers
Joe Buck , John Smoltz , Ken Rosenthal , and Tom Verducci ( Fox ) Rolando Nichols , Carlos Álvarez , Edgar Gonzalez ,
</item 4>

<item 5>
2017 World Series
Sports Business Journal . Retrieved October 4 , 2017 . External links ( edit ) 2017 World Series at Baseball Reference 2017 Major League Baseball postseason
Part of the 2017 Major League Baseball season World Series American League Championship Series National League Championship Series American League Division
Series National League Division Series American League Wild Card Game National League Wild Card Game American League teams Boston Red Sox Cleveland Indians
Houston Astros Minnesota Twins New York Yankees National League teams Arizona Diamondbacks Chicago Cubs Colorado Rockies Los Angeles Dodgers Washington
Nationals World Series No World Series was held in 1904 because the NL champions refused to participate ; no World Series was held in 1994 due to a players '
strike . 1900s -- 1910s 1900 1901 1902 1903 1904 1905 1906 1907 1908 1909 1910 1911 1912 1913 1914 1915 1916 1917 1918 1919 1920s -- 1930s 1920 1921 1922 1923
1924 1925 1926 1927 1928 1929
</item 5>

<item 6>
2017 World Series
The 113th edition of the World Series , it was played between October 24 and November 1 . The series was a best - of - seven playoff between the National League
( NL ) champion Los Angeles Dodgers and the American League ( AL ) champion Houston Astros . It was sponsored by the internet television service YouTube TV and
officially known as the World Series presented by YouTube TV . The Astros defeated the Dodgers , four games to three , to win their first World Series in
franchise history , and the first team from Texas to do so . Both teams set a World Series record with a combined total of 25 home runs throughout the entire
series , including a team record 15 home runs by the Astros , and hit a combined total of eight home runs in Game 2 to set the single game World Series mark .
Houston outfielder George Springer was named as the World Series Most Valuable Player ( MVP ) after hitting five home runs in the series to tie a World Series
record with Reggie Jackson in 1977 and Dodgers ' 2017 World Series representative Chase Utley in 2009 when he played for the Philadelphia Phillies .
</item 6>

<item 7>
2017 World Series
1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series
champions Most Valuable Players Starting pitchers Babe Ruth Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise )
Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7 1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series
Category : World Series Fox Major League Baseball Related programs The Cheap Seats ( 2010 -- 2011 ) MLB Whiparound Major League Baseball Game of the Week
Thursday Night Baseball ( 1997 -- 2001 ) This Week in Baseball ( 2000 -- 2011 ) Related articles DirecTV N3D FoxBox FoxTrax Scooter Television contracts ( cable
) MLB Network World Series television ratings National coverage Fox ( 1996 -- present ) FS1 ( 2014 -- present
</item 7>

<item 8>
2017 World Series
The series is officially known as the 2017 World Series Presented by YouTube TV . This sponsorship includes logo branding in - stadium and on official digital
properties , as well as commercial inventory during Fox 's telecasts of the games . See also ( edit ) Baseball portal Los Angeles portal Houston portal 2017
Korean Series 2017 Japan Series Notes ( edit ) Jump up ^ Except 1904 and 1994 , when the World Series was not held References ( edit ) Jump up ^ Macklin ,
Oliver ( October 24 , 2017 ) . `` Davis named chief of World Series ump crew '' . Major League Baseball . Retrieved October 24 , 2017 . Jump up ^ Spangler ,
Todd ( October 4 , 2017 ) . `` YouTube TV Swings for 2017 World Series Hits in MLB Pact '' . Variety . Retrieved November 2 , 2017 . We are just a week away
from the World Series presented by YouTube TV ^ Jump up to : Kramer , Daniel ( October 13 , 2017 ) . `` Overview of World Series home - field advantage '' .
Major League Baseball .
</item 8>

<item 9>
The text is mostly historical navigation for MLB postseason and World Series pages, including the 2017 World Series and links to Baseball Reference and Sports
Business Journal. It lists 2017 playoff teams: Red Sox, Indians, Astros, Twins, Yankees, Diamondbacks, Cubs, Rockies, Dodgers, and Nationals. It notes no World
Series was held in 1904 or 1994 and also contains references to the 2005 World Series and Stanley Cup playoffs.
</item 9>

<item 10>
2005 World Series
1977 1978 1979 1980s -- 1990s 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000s -- 2010s 2000 2001 2002 2003 2005 2006
2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 See also Pre-World Series champions World Series champions Most Valuable Players Starting pitchers Babe Ruth
Award Commissioner 's Trophy World Series ring Appearances Streaks Droughts Series ( by franchise ) Broadcasters TV ratings ALCS NLCS ALDS NLDS ALWC NLWC Game 7
1989 Loma Prieta earthquake Dauvray Cup Temple Cup Chronicle - Telegraph Cup Book : World Series Category : World Series Chicago White Sox Based in Chicago ,
</item 10>
=== END ITEM id=e6abad7fc50a58ef ===

=== ITEM id=6276e6f4a5a63b52 ===
Gold sentences and facts:
Sentence 0: A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called `` perfecting the security interest '' in the property , and this type of loan is a secured loan . A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit documents and the lease , respectively , and not the financing statement .
  Fact 0.0: A UCC-1 financing statement is a legal form.
  Fact 0.1: UCC-1 is an abbreviation for Uniform Commercial Code-1.
  Fact 0.2: A creditor files a UCC-1 financing statement to give notice that it has or may have an interest in the personal property of a debtor.
  Fact 0.3: A debtor is a person who owes a debt to the creditor, as typically specified in the agreement creating the debt.
  Fact 0.4: A financing statement is filed to perfect a creditor's security interest.
  Fact 0.5: The filing gives public notice that there is a right to take possession of and sell certain assets for repayment of a specific debt, with a certain priority.
  Fact 0.6: Such notices of sale are often found in local newspapers.
  Fact 0.7: Once the form has been filed, the creditor establishes a relative priority with other creditors of the debtor.
  Fact 0.8: Perfecting the security interest is also called perfecting the security interest in the property.
  Fact 0.9: This type of loan is a secured loan.
  Fact 0.10: A financing statement may also be filed in the real estate records by a lessor of fixtures.
  Fact 0.11: The real estate filing establishes the lessor's priority against a holder of a mortgage or other lien on the real property.
  Fact 0.12: The creditor's rights against the debtor are based on the credit documents, not the financing statement.
  Fact 0.13: The lessor's rights against the lessee are based on the lease, not the financing statement.
Sentence 1: The financing statement is generally filed with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which is tasked with overseeing business organizations and activities , including receipt of financing statements . However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures . In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where third parties are most likely to search for such record .
  Fact 1.0: The financing statement is generally filed with the office of the state secretary of state.
  Fact 1.1: The financing statement is generally filed in the state where the debtor is located.
  Fact 1.2: For an individual, the state where the debtor is located is the state where the debtor resides.
  Fact 1.3: For most kinds of business organizations, the state where the debtor is located is the state of incorporation or organization.
  Fact 1.4: Many states have a state agency that operates under the secretary of state.
  Fact 1.5: That state agency is tasked with overseeing business organizations and activities, including receipt of financing statements.
  Fact 1.6: An exception exists if the collateral is tied to a particular piece of real property, such as timber, mineral rights, or fixtures.
  Fact 1.7: In that exception case, the filing must be made in the county where the property is located.
  Fact 1.8: In that exception case, the filing is usually made in the recording office or county court.
  Fact 1.9: Third parties are most likely to search for such a record in that county.

Retrieved items:
<item 1>
Ownership
A structured group duly constituted as an entity under law may still not protect members from being personally liable for each other 's actions . Court
decisions against the entity itself may give rise to unlimited personal liability for each and every member . An example of this situation is a professional
partnership ( e.g. law practice ) in some jurisdictions . Thus , being a partner or owner in a group may give little advantage in terms of share ownership while
producing a lot of risk to the partner , owner or participant . Sharing gains ( edit ) At the end of each financial year , accounting rules determine a surplus
or profit , which may be retained inside the entity or distributed among owners according to the initial setup intent when the entity was created . For public
corporations , common shareholders have no right to receive any of the profit . Entities with a member focus will give financial surplus back to members
according to the volume of financial activity that the participating member generated for the entity . Examples of this are producer cooperatives , buyer
cooperatives and participating whole life policyholders in both mutual and share - capital insurance companies . Entities with share voting rights that depend
on financial capital distribute surplus among shareholders without regard to any other contribution to the entity .
</item 1>

<item 2>
Ownership
A legal entity is a legal construct through which the law allows a group of natural persons to act as if it were an individual for certain purposes . Some duly
incorporated entities may not be owned by individuals nor by other entities ; they exist without being owned once they are created . Not being owned , they can
not be bought and sold . Mutual life insurance companies , credit unions , foundations and cooperatives , not for profit organizations , and public corporations
are examples of this . No person can purchase the company , as their ownership is not legally available for sale , neither as shares nor as a single whole .
Intellectual property ( edit ) Main article : Intellectual property Intellectual property ( IP ) refers to a legal entitlement which sometimes attaches to the
expressed form of an idea , or to some other intangible subject matter . This legal entitlement generally enables its holder to exercise exclusive rights of use
in relation to the subject matter of the IP . The term intellectual property reflects the idea that this subject matter is the product of the mind or the
intellect , and that IP rights may be protected at law in the same way as any other form of property .
</item 2>

<item 3>
UCC-1 financing statement
However , an exception exists if the collateral is something that is tied to a particular piece of real property , such as timber , mineral rights , or fixtures
. In that case , the filing must be made in the county where the property is located , usually in the recording office or county court , because that is where
third parties are most likely to search for such record . In the case of a loan secured by personal property collateral , the filing of a financing statement
gives notice of a lien against the property so that other lenders or buyers of the personal property will know of the security interest . In the case of a
filing of a financing statement by a lessor of fixtures , the filing of the financing statement gives notice of the lessor 's interests to others who acquire an
interest in the real property and related fixtures . The financing statement does not create a lien nor does it create any additional rights against a lessee in
favor of a lessor , the filing of a financing statement just gives notice of whatever rights the creditor or lessor have under their loan documents or lease ,
respectively .
</item 3>

<item 4>
IRS penalties
Jump up ^ 26 USC 6651 ( a ) ( 1 ) . Jump up ^ IRC § 6651 ( a ) ( 1 ) Jump up ^ IRC § 6651 ( a ) ( 2 ) Jump up ^ `` 1040 ( 2015 ) '' . Internal Revenue Service .
Retrieved June 28 , 2016 . Jump up ^ 26 USC 6662 . Jump up ^ 26 USC 6721 . Jump up ^ 26 CFR 301 - 6721 - 1 ( a ) ( 2 ) ( ii ) . Jump up ^ 26 USC 6698 . Jump up
^ 26 USC 6699 . Jump up ^ Technically , the Form 1120S is an income tax return , not an information return . Jump up ^ 26 USC 3101 - 3406 . Jump up ^ 26 USC
6672 . Jump up ^ 26 USC 3504 . Jump up ^ See , e.g. , United States v. St. Clair , 45 Amer . Fed . Tax Rep.
</item 4>

<item 5>
UCC-1 financing statement
A financing statement may also be filed in the real estate records by a lessor of fixtures to establish the priority of the lessor 's rights against a holder of
a mortgage or other lien on the real property . The creditor 's rights against the debtor and the lessor 's rights against the lessee are based on the credit
documents and the lease , respectively , and not the financing statement . Pursuant to the standards set forth in the UCC , at 9 - 503 and 9 - 504 , the
financing statement need only contain three pieces of information : the debtor 's name and address the creditor 's name and address an indication of the
collateral , `` whether or not it is specific , if it reasonably identifies what is described . '' ( UCC 9 - 108 ) The financing statement is generally filed
with the office of the state secretary of state , in the state where the debtor is located - for an individual , the state where the debtor resides , for most
kinds of business organizations the state of incorporation or organization . Many states have a state agency that operates under the secretary of state , which
is tasked with overseeing business organizations and activities , including receipt of financing statements .
</item 5>

<item 6>
United Kingdom corporation tax
Individuals could set off the tax credit against their income tax liability . On introduction , ACT was set at 30 % of the gross dividend ( the actual amount
paid plus the tax credit ) . If a company made a £ 70 dividend payment to an individual , the company would pay £ 30 of advance corporation tax . The
shareholder would receive the £ 70 cash payment , plus a tax credit of £ 30 ; thus , the individual would be deemed to have earned £ 100 , and to have already
paid tax of £ 30 on it . The ACT paid by the company would be deductible against its final `` mainstream '' corporation tax bill . To the extent that the
individual 's tax on the dividend was less than the tax credit -- for example , if his income was too low to pay tax ( below £ 595 in 1973 -- 1974 ) -- he would
be able to reclaim some or all of the £ 30 tax paid by the company .
</item 6>

<item 7>
IRS penalties
Clair , 45 Amer . Fed . Tax Rep. 2d 80 - 1528 , 80 - 1 U.S. Tax Cas . ( CCH ) ¶ 9364 , ( N.D. Texas 1980 ) , in which the court upheld seizure of the taxpayer
's personal residence for failure to pay over withholding taxes payable by the company . Jump up ^ 26 USC 6038 . Jump up ^ 26 USC 6038A . Jump up ^ 26 USC 6689
. Jump up ^ 26 USC 6677 . Jump up ^ 26 USC 6038B . Jump up ^ Subtitle D of 26 USC . Jump up ^ Manufacturers Taxes , IRS Jump up ^ 26 USC 4971 - 4980G Jump up ^
26 USC 4911 , 4912 and 26 USC 4940 - 4948 . Jump up ^ 26 USC 7301 - 7328 . Jump up ^ 26 USC 7201 - 7217 . Jump up ^ 26 USC 6702 26 USC 6673 .
</item 7>

<item 8>
IRS penalties
U.S. corporations more than 25 % owned , directly or indirectly , by foreign persons must file Form 5472 to report such ownership and all transactions with
related parties . Failure to timely file carries a $10,000 penalty per required form . This penalty may be increased by $10,000 per month per form for continued
failure to file . In addition , taxpayers who fail to report changes in foreign taxes used as credits against Federal income tax may be subject to penalties .
U.S. citizen or resident taxpayers ( including entities ) who are beneficiaries of a foreign trust or make transfers of property to a foreign trust must report
information about the transfer and the trust or corporation . Failure to timely report on Form 3520 or Form 3520 - A may result in penalties of up to 35 % .
Similar transferors to foreign corporations failing to file Form 926 may face penalties of 10 % of the value of the transfer , up to $100,000 . Excise taxes as
penalties ( edit ) Federal excise taxes are imposed on a variety of goods and services . Some of these taxes require purchase of tax stamps or other evidence of
advance payment of tax .
</item 8>

<item 9>
UCC-1 financing statement
References ( edit ) Jump up ^ http://www.findlegalforms.com/articles/form-encyclopedia/what-is-a-ucc1-financing-statement Jump up ^
http://biztaxlaw.about.com/od/glossaryu/g/ucc1statement.htm Retrieved from ``
https://en.wikipedia.org/w/index.php?title=UCC-1_financing_statement&oldid=803697764 '' Categories : Uniform Commercial Code Legal terminology Hidden categories
: Articles needing additional references from October 2014 All articles needing additional references Talk About Wikipedia Add links This page was last edited
on 4 October 2017 , at 02 : 00 . About Wikipedia
</item 9>

<item 10>
UCC-1 financing statement
UCC - 1 financing statement - wikipedia UCC - 1 financing statement Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2014 ) ( Learn how and
when to remove this template message ) A UCC - 1 financing statement ( an abbreviation for Uniform Commercial Code - 1 ) is a legal form that a creditor files
to give notice that it has or may have an interest in the personal property of a debtor ( a person who owes a debt to the creditor as typically specified in the
agreement creating the debt ) . This form is filed in order to `` perfect '' a creditor 's security interest by giving public notice that there is a right to
take possession of and sell certain assets for repayment of a specific debt with a certain priority . Such notices of sale are often found in the local
newspapers . Once the form has been filed , the creditor establishes a relative priority with other creditors of the debtor . This process is also called ``
perfecting the security interest '' in the property , and this type of loan is a secured loan .
</item 10>
=== END ITEM id=6276e6f4a5a63b52 ===

=== ITEM id=ac45bd7d764f1724 ===
Gold sentences and facts:
Sentence 0: A muscle relaxant is a drug that affects skeletal muscle function and decreases the muscle tone . It may be used to alleviate symptoms such as muscle spasms , pain , and hyperreflexia . The term `` muscle relaxant '' is used to refer to two major therapeutic groups : neuromuscular blockers and spasmolytics . Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate and have no central nervous system ( CNS ) activity . They are often used during surgical procedures and in intensive care and emergency medicine to cause temporary paralysis . Spasmolytics , also known as `` centrally acting '' muscle relaxants , are used to alleviate musculoskeletal pain and spasms and to reduce spasticity in a variety of neurological conditions . While both neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants , the term is commonly used to refer to spasmolytics only .
  Fact 0.0: A muscle relaxant is a drug that affects skeletal muscle function.
  Fact 0.1: A muscle relaxant decreases muscle tone.
  Fact 0.2: A muscle relaxant may be used to alleviate muscle spasms.
  Fact 0.3: A muscle relaxant may be used to alleviate pain.
  Fact 0.4: A muscle relaxant may be used to alleviate hyperreflexia.
  Fact 0.5: The term 'muscle relaxant' is used to refer to two major therapeutic groups.
  Fact 0.6: The two major therapeutic groups referred to by 'muscle relaxant' are neuromuscular blockers and spasmolytics.
  Fact 0.7: Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate.
  Fact 0.8: Neuromuscular blockers have no central nervous system (CNS) activity.
  Fact 0.9: Neuromuscular blockers are often used during surgical procedures.
  Fact 0.10: Neuromuscular blockers are often used in intensive care and emergency medicine.
  Fact 0.11: Neuromuscular blockers are used to cause temporary paralysis.
  Fact 0.12: Spasmolytics are also known as 'centrally acting' muscle relaxants.
  Fact 0.13: Spasmolytics are used to alleviate musculoskeletal pain and spasms.
  Fact 0.14: Spasmolytics are used to reduce spasticity in a variety of neurological conditions.
  Fact 0.15: Neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants.
  Fact 0.16: The term 'muscle relaxant' is commonly used to refer to spasmolytics only.

Retrieved items:
<item 1>
Muscle relaxant
Muscle relaxant - wikipedia Muscle relaxant Jump to : navigation , search This article is about skeletal muscle relaxants . For information on smooth muscle
relaxants , see Antispasmodic . A muscle relaxant is a drug that affects skeletal muscle function and decreases the muscle tone . It may be used to alleviate
symptoms such as muscle spasms , pain , and hyperreflexia . The term `` muscle relaxant '' is used to refer to two major therapeutic groups : neuromuscular
blockers and spasmolytics . Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate and have no central nervous system ( CNS
) activity . They are often used during surgical procedures and in intensive care and emergency medicine to cause temporary paralysis . Spasmolytics , also
known as `` centrally acting '' muscle relaxants , are used to alleviate musculoskeletal pain and spasms and to reduce spasticity in a variety of neurological
conditions . While both neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants , the term is commonly used to refer to
spasmolytics only .
</item 1>

<item 2>
Muscle relaxant
PMID 17909209 . ^ Jump up to : van Tulder MW , Touray T , Furlan AD , Solway S , Bouter LM ( 2003 ) . Van Tulder , Maurits W , ed . `` Muscle relaxants for
non-specific low back pain '' . Cochrane Database Syst Rev ( 2 ) : CD004252 . doi : 10.1002 / 14651858. CD004252 . PMID 12804507 . CS1 maint : Multiple names :
authors list ( link ) ^ Jump up to : Beebe FA , Barkin RL , Barkin S ( 2005 ) . `` A clinical and pharmacologic review of skeletal muscle relaxants for
musculoskeletal conditions '' . Am J Ther. 12 ( 2 ) : 151 -- 71 . doi : 10.1097 / 01. mjt. 0000134786.50087. d8 . PMID 15767833 . Jump up ^ See , S. ; Ginzburg
, R. ( Feb 2008 ) . `` Skeletal muscle relaxants '' .
</item 2>

<item 3>
Muscle relaxant
dantrolene ) , so this name is inaccurate . Most sources still use the term `` centrally acting muscle relaxant '' . According to MeSH , dantrolene is usually
classified as a centrally acting muscle relaxant . The World Health Organization , in its ATC , uses the term `` centrally acting agents '' , but adds a
distinct category of `` directly acting agents '' , for dantrolene . Use of this terminology dates back to at least 1973 . The term `` spasmolytic '' is also
considered a synonym for antispasmodic . Clinical use ( edit ) Spasmolytics such as carisoprodol , cyclobenzaprine , metaxalone , and methocarbamol are commonly
prescribed for low back pain or neck pain , fibromyalgia , tension headaches and myofascial pain syndrome . However , they are not recommended as first - line
agents ; in acute low back pain , they are not more effective than paracetamol or nonsteroidal anti-inflammatory drugs ( NSAIDs ) , and in fibromyalgia they are
not more effective than antidepressants .
</item 3>

<item 4>
Muscle relaxant
Contents ( hide ) 1 History 2 Neuromuscular blockers 3 Spasmolytics 3.1 Terminology 3.2 Clinical use 3.3 Mechanism 4 Side effects 5 See also 6 References 7
External links History ( edit ) The earliest known use of muscle relaxant drugs was by natives of the Amazon Basin in South America who used poison - tipped
arrows that produced death by skeletal muscle paralysis . This was first documented in the 16th century , when European explorers encountered it . This poison ,
known today as curare , led to some of the earliest scientific studies in pharmacology . Its active ingredient , tubocurarine , as well as many synthetic
derivatives , played a significant role in scientific experiments to determine the function of acetylcholine in neuromuscular transmission . By 1943 ,
neuromuscular blocking drugs became established as muscle relaxants in the practice of anesthesia and surgery . The U.S. Food and Drug Administration ( FDA )
approved the use of carisoprodol in 1959 , metaxalone in August 1962 , and cyclobenzaprine in August 1977 .
</item 4>

<item 5>
Muscle relaxant
( Feb 2008 ) . `` Skeletal muscle relaxants '' . Pharmacotherapy. 28 ( 2 ) : 207 -- 13 . doi : 10.1592 / phco. 28.2. 207 . PMID 18225966 . Jump up ^ `` When Are
Muscle Relaxers Prescribed For Arthritis Patients ? '' . Retrieved 2010 - 07 - 09 . Jump up ^ MedlinePlus Encyclopedia Tension Headache Jump up ^ `` Muscle
Relaxants '' . Retrieved 2010 - 07 - 09 . Jump up ^ Rang , H.P. ; Dale , M.M. ( 1991 ) . `` Drugs Used in Treating Motor Disorders '' . Pharmacology ( 2nd ed .
) . Churchill Livingston . pp. 684 -- 705 . ISBN 044304483X . Jump up ^ Standaert , D.G. ; Young , A.B. ( 2001 ) . `` Treatment Of Central Nervous System
Degerative Disorders '' . In Goodman , L.S.
</item 5>

<item 6>
Muscle relaxant
Chemical diagram of pancuronium , with red lines indicating the two acetylcholine `` molecules '' in the structure Spasmolytics ( edit ) Main article :
antispasmodic A view of the spinal cord and skeletal muscle showing the action of various muscle relaxants -- black lines ending in arrow heads represent
chemicals or actions that enhance the target of the lines , blue lines ending in squares represent chemicals or actions that inhibition the target of the line
The generation of the neuronal signals in motor neurons that cause muscle contractions are dependent on the balance of synaptic excitation and inhibition the
motor neuron receives . Spasmolytic agents generally work by either enhancing the level of inhibition , or reducing the level of excitation . Inhibition is
enhanced by mimicking or enhancing the actions of endogenous inhibitory substances , such as GABA . Terminology ( edit ) Because they may act at the level of
the cortex , brain stem or spinal cord , or all three areas , they have traditionally been referred to as `` centrally acting '' muscle relaxants . However , it
is now known not every agent in this class has CNS activity ( e.g. dantrolene ) , so this name is inaccurate .
</item 6>

<item 7>
Muscle relaxant
Retrieved 2010 - 07 - 09 . External links ( edit ) Skeletal Muscle Relaxants at the US National Library of Medicine Medical Subject Headings ( MeSH ) Skeletal
muscle relaxants ( M03 ) Peripherally acting ( primarily antinicotinic , NMJ block ) Non-depolarizing Curare alkaloids Alcuronium Dimethyltubocurarine
Tubocurarine 4 ° ammonium agents ultra-short duration : Gantacurium short duration : Mivacurium Chandonium intermediate duration : Atracurium Cisatracurium
Fazadinium Rocuronium Vecuronium long duration : Doxacurium Dimethyltubocurarine Pancuronium Pipecuronium Laudexium Gallamine unsorted : Hexafluronium (
Hexafluorenium ) Depolarizing Choline derivatives : Suxamethonium ( Succinylcholine ) Polyalkylene derivatives : Hexamethonium ACh release inhibitors Botulinum
toxin Centrally acting Carbamic acid esters Carisoprodol Cyclarbamate Difebarbamate Febarbamate Meprobamate Phenprobamate Styramate Tybamate
</item 7>

<item 8>
Muscle relaxant
Mayo Clinic . Last Updated : April 1 , 2007 . Retrieved on September 19 , 2007 . ^ Jump up to : Miller , R.D. ( 1998 ) . `` Skeletal Muscle Relaxants '' . In
Katzung , B.G. Basic & Clinical Pharmacology ( 7th ed . ) . Appleton & Lange . pp. 434 -- 449 . ISBN 0 - 8385 - 0565 - 1 . Jump up ^ Bowman WC ( January 2006 )
. `` Neuromuscular block '' . Br. J. Pharmacol. 147 Suppl 1 : S277 -- 86 . doi : 10.1038 / sj. bjp. 0706404 . PMC 1760749 . PMID 16402115 . Jump up ^ `` Brief
History '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Brief History '' . Retrieved 2010 - 07 - 09 . Jump up ^ Craig , C.R. ; Stitzel , R.E.
</item 8>

<item 9>
Muscle relaxant
Other skeletal muscle relaxants of that type used around the world come from a number of drug categories and other drugs used primarily for this indication
include orphenadrine ( anticholinergic ) , chlorzoxazone , tizanidine ( clonidine relative ) , diazepam , tetrazepam and other benzodiazepines , mephenoxalone ,
methocarbamol , dantrolene , baclofen , Drugs once but no longer or very rarely used to relax skeletal muscles include meprobamate , barbiturates , methaqualone
, glutethimide and the like ; some subcategories of opioids have muscle relaxant properties , and some are marketed in combination drugs with skeletal and / or
smooth muscle relaxants such as whole opium products , some ketobemidone , piritramide and fentanyl preparations and Equagesic . Neuromuscular blockers ( edit )
Detailed view of a neuromuscular junction : 1 . Presynaptic terminal 2 . Sarcolemma 3 . Synaptic vesicle 4 . Nicotinic acetylcholine receptor 5 .
</item 9>

<item 10>
Muscle relaxant
Eur. J. Neurosci. 12 ( 2 ) : 701 -- 14 . doi : 10.1046 / j. 1460 - 9568.2000. 00955. x . PMID 10712650 . Jump up ^ Young RR , Wiegner AW ( 1987 ) . ``
Spasticity '' . Clin. Orthop . Relat . Res. 219 : 50 -- 62 . PMID 3581584 . Jump up ^ `` The Effects of Muscle Relaxant Drugs Are Powerful '' . Retrieved 2010 -
07 - 09 . Jump up ^ `` Cyclobenzaprine - Oral '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Carisoprodolx '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Side
Effects of Metaxalone -- for the Consumer '' . Retrieved 2010 - 07 - 09 . Jump up ^ `` Precautions '' . Retrieved 2010 - 07 - 09 .
</item 10>
=== END ITEM id=ac45bd7d764f1724 ===

=== ITEM id=9f0e159d91d2e019 ===
Gold sentences and facts:
Sentence 0: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 )
  Fact 0.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 0.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 0.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 0.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 0.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 0.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 0.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 0.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 0.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 0.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 0.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 0.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 0.12: Bart Cummings won the Melbourne Cup in 2008.
Sentence 1: 12 -- Bart Cummings ( 1965 , 1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman ( 1989 , 1992 , 1995 , 2004 , 2005 )
  Fact 1.0: Bart Cummings has 12 Melbourne Cup wins.
  Fact 1.1: Bart Cummings won the Melbourne Cup in 1965.
  Fact 1.2: Bart Cummings won the Melbourne Cup in 1966.
  Fact 1.3: Bart Cummings won the Melbourne Cup in 1967.
  Fact 1.4: Bart Cummings won the Melbourne Cup in 1974.
  Fact 1.5: Bart Cummings won the Melbourne Cup in 1975.
  Fact 1.6: Bart Cummings won the Melbourne Cup in 1977.
  Fact 1.7: Bart Cummings won the Melbourne Cup in 1979.
  Fact 1.8: Bart Cummings won the Melbourne Cup in 1990.
  Fact 1.9: Bart Cummings won the Melbourne Cup in 1991.
  Fact 1.10: Bart Cummings won the Melbourne Cup in 1996.
  Fact 1.11: Bart Cummings won the Melbourne Cup in 1999.
  Fact 1.12: Bart Cummings won the Melbourne Cup in 2008.
  Fact 1.13: Etienne de Mestre has 5 Melbourne Cup wins.
  Fact 1.14: Etienne de Mestre won the Melbourne Cup in 1861.
  Fact 1.15: Etienne de Mestre won the Melbourne Cup in 1862.
  Fact 1.16: Etienne de Mestre won the Melbourne Cup in 1867.
  Fact 1.17: Etienne de Mestre won the Melbourne Cup in 1877.
  Fact 1.18: Etienne de Mestre won the Melbourne Cup in 1878.
  Fact 1.19: Lee Freedman has 5 Melbourne Cup wins.
  Fact 1.20: Lee Freedman won the Melbourne Cup in 1989.
  Fact 1.21: Lee Freedman won the Melbourne Cup in 1992.
  Fact 1.22: Lee Freedman won the Melbourne Cup in 1995.
  Fact 1.23: Lee Freedman won the Melbourne Cup in 2004.
  Fact 1.24: Lee Freedman won the Melbourne Cup in 2005.
Sentence 2: 1877 is also the year that the trainer Etienne de Mestre won his fourth Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell ( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who , though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth Melbourne Cup in 1975 . Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 .
  Fact 2.0: In 1877, the trainer Etienne de Mestre won his fourth Melbourne Cup.
  Fact 2.1: In 1877, Etienne de Mestre won with Chester.
  Fact 2.2: Chester was owned by Hon. James White.
  Fact 2.3: In 1878, as in previous years, Etienne de Mestre fielded more than one horse.
  Fact 2.4: Etienne de Mestre entered the favourite Firebell in 1878.
  Fact 2.5: Firebell was owned by W.S. Cox.
  Fact 2.6: Firebell finished last in 1878.
  Fact 2.7: Etienne de Mestre entered Chester in 1878.
  Fact 2.8: Chester was owned by Hon. James White.
  Fact 2.9: Chester was the previous year's winner.
  Fact 2.10: Chester fell in 1878.
  Fact 2.11: Etienne de Mestre entered Calamia in 1878.
  Fact 2.12: Calamia was owned by Etienne de Mestre.
  Fact 2.13: Calamia was less fancied.
  Fact 2.14: Calamia won easily by two lengths in 1878.
  Fact 2.15: First prize in 1878 was £1,790.
  Fact 2.16: The crowd in 1878 was 80,000.
  Fact 2.17: There were 30 starters in 1878.
  Fact 2.18: De Mestre's 1878 win with Calamia brought to 5 the number of Melbourne Cups he had won.
  Fact 2.19: De Mestre's record of 5 Melbourne Cups was not matched for nearly 100 years.
  Fact 2.20: Bart Cummings won his fifth Melbourne Cup in 1975.
  Fact 2.21: Bart Cummings is regarded as the best Australian horse trainer of all time.
  Fact 2.22: Bart Cummings went on to win 12 Melbourne Cups, up to 2008.

Retrieved items:
<item 1>
Melbourne Cup
The first jockey of Indigenous heritage to ride a Melbourne Cup winner was Frank Reys in 1973 on Gala Supreme , who had a Filipino father and a half -
Aboriginal mother . Recent years ( edit ) Wikinews has related news : Delta Blues wins 2006 Melbourne Cup Makybe Diva wins the Melbourne Cup Efficient wins
Melbourne Cup Viewed wins the 2008 Melbourne Cup The race has undergone several alterations in recent years , the most visible being the entry of many foreign -
trained horses . Most have failed to cope with the conditions ; the three successful `` foreign raids '' include two by Irish trainer Dermot K. Weld successful
in 1993 and 2002 , and one in 2006 by Katsumi Yoshida of Japan 's renowned Yoshida racing and breeding family . The attraction for foreigners to compete was ,
primarily , the low - profile change to the new `` quality handicap '' weighting system . The 1910 Melbourne Cup was won by Comedy King , the first foreign bred
horse to do so .
</item 1>

<item 2>
Melbourne Cup
Bart Cummings , regarded as the best Australian horse trainer of all time , went on to win 12 Melbourne Cups to 2008 . Martini - Henry , the 1883 Melbourne Cup
winner In 1883 , the hardy New Zealand bred , Martini - Henry won the VRC Derby , the Melbourne Cup and on the following Monday retained his undefeated record
by winning Mares ' Produce Stakes . Phar Lap winning the Melbourne Cup Race from Second Wind and Shadow King on 5 November 1930 . Phar Lap , the most famous
horse in the world of his day , won the 1930 Melbourne Cup at 11 / 8 odds on , the shortest priced favourite in the history of the race . He had to be hidden
away at Geelong before the race after an attempt was made to shoot him and only emerged an hour before the race time of the Cup . Phar Lap also competed in 1929
and 1931 , but came 3rd and 8th respectively , despite heavy favouritism in both years . There are a few legends of the first Aboriginal jockey to ride a
Melbourne Cup .
</item 2>

<item 3>
Melbourne Cup
Perhaps they should have kept their money , however . The outsider Briseis comfortably won by 1 length in the biggest field of all time , and in an extremely
good time , so it is unlikely that the horses who perished could have beaten her . 1877 is also the year that the trainer Etienne de Mestre won his fourth
Melbourne Cup with Chester owned by Hon . James White . In 1878 , as in previous years De Mestre fielded more than one horse . He entered the favourite Firebell
( owned by W.S. Cox ) who finished last , Chester ( owned by Hon . James White ) the previous year 's winner who fell , and Calamia ( owned by de Mestre ) who ,
though less fancied , won easily by two lengths . First prize was £ 1,790 , the crowd was 80,000 and there were 30 starters . De Mestre 's 1878 win with Calamia
brought to 5 the number of Melbourne Cups he had won . This record was not to be matched for nearly 100 years when the trainer Bart Cummings won his fifth
Melbourne Cup in 1975 .
</item 3>

<item 4>
Melbourne Cup
The race was won by the George Hanlon trained 40 - 1 outsider Piping Lane ridden by John Letts and carrying only 48 kg . 1985 -- The first sponsored Melbourne
Cup , and the first million dollar Cup , with $650,000 for the winner . 1987 -- First female jockey to ride in the cup was Maree Lyndon on Argonaut Style .
Dunaden : 2011 Melbourne Cup winner , painted by Charles Church 1993 -- The Dermot Weld trained Irish gelding Vintage Crop became the first Northern hemisphere
trained horse to win the Melbourne Cup and the first Irish horse . 2001 -- Sheila Laxon , was the first woman trainer to officially win the Melbourne Cup .
However , Mrs. A. McDonald ( 1938 ) with Catalogue was really the first woman trainer to win . Women then could not be registered as trainers in Australia , and
it was her husband who was the registered trainer . Mrs. McDonald 's win was as a female trainer of a female owned horse . 2003 -- First Australian female
jockey to ride in the cup was Clare Lindop on Debben .
</item 4>

<item 5>
Melbourne Cup
1966 , 1967 , 1974 , 1975 , 1977 , 1979 , 1990 , 1991 , 1996 , 1999 , 2008 ) 5 -- Etienne de Mestre ( 1861 , 1862 , 1867 , 1877 , 1878 ) 5 -- Lee Freedman (
1989 , 1992 , 1995 , 2004 , 2005 ) Most wins by an owner ( edit ) 6 -- Lloyd Williams ( 1981 , 1985 , 2007 , 2012 , 2016 , 2017 ) 4 -- John Tait ( 1866 , 1868 ,
1871 , 1872 ) 4 -- Etienne de Mestre ( 1861 * , 1862 * , 1867 * , 1878 ) * The winners of these years were leased . 4 -- Dato Tan Chin Nam ( 1974 , 1975 , 1996
, 2008 ) Other records ( edit ) Record winning time -- 3.16.
</item 5>

<item 6>
Melbourne Cup
Most runners -- 39 ( 1890 ) Fewest runners -- 7 ( 1863 ) Most attempts -- Shadow King made six attempts to win the cup in seven years between 1929 and 1935 . He
ran 6th , 3rd , 2nd , 3rd , 2nd and 4th . Attendance ( edit ) The horse show as well as fashion show of Melbourne Cup takes place on the lawn The event is one
of the most popular spectator events in Australia , with sometimes over 110,000 people , some dressed in traditional formal raceday wear and others in all
manner of exotic and amusing costumes , attending the race . The record crowd was 122,736 in 2003 . The 1926 running of the Cup was the first time the 100,000
mark had been passed . Today the record at Flemington is held by the 2006 Victoria Derby when almost 130,000 attended . In 2007 , a limit was placed on the
Spring Carnival attendance at Flemington Racecourse and race - goers are now required to pre-purchase tickets .
</item 6>

<item 7>
Melbourne Cup
Subsequent foreign bred horses to win Cup were Backwood 1924 ; Phar Lap 1930 ; Wotan 1936 Belldale Ball 1980 ; At Talaq 1986 ; Kingston Rule 1990 ; Vintage Crop
1993 ; Jeune 1994 ; Media Puzzle 2002 ; Makybe Diva 2003 , 2004 , 2005 ; Americain 2010 and Dunaden 2011 . The 1938 Melbourne Cup was won by trainer Mrs. Allan
McDonald , who conditioned Catalogue . Mrs McDonald was a successful trainer in New Zealand , however , at the time women were not allowed to compete as
trainers in Australia so her husband 's name was officially recorded as the winning trainer . The 2001 edition was won by New Zealand mare Ethereal , trained by
Sheila Laxon , the first woman to formally train a Melbourne Cup winner . She also won the Caulfield Cup , a 2,400 metre race also held in Melbourne , and
therefore has won the `` Cups Double '' .
</item 7>

<item 8>
Melbourne Cup
2003 -- The first Melbourne Cup Tour was conducted around Australia , and the biggest crowd , of 122,736 , is recorded at Flemington . 2005 -- Makybe Diva
became the only horse so far to win the Melbourne Cup three times . 2008 -- The `` Cup King '' , Bart Cummings , took his 12th win in the Melbourne Cup with
Viewed 2010 - 150th anniversary . Americain becomes the first French - trained horse to win the race , and Gerald Mosse the first French jockey . 2011 - French
based horse Dunaden wins from Red Cadeaux in the closest finish ever . 2012 - The first seven finishers in the race were bred in Ireland . 2013 - Trainer Gai
Waterhouse became the first Australian female trainer to train the winner of the race when Fiorente won . Her father , legendary trainer Tommy J. Smith was also
a Melbourne Cup Winning trainer ( Toparoa in 1955 and Just A Dash in 1981 ) . 2014 - Two horses die due to racing in the 2014 Melbourne Cup .
</item 8>

<item 9>
Melbourne Cup
1 Melbourne Cup Tour 3 History 3.1 Early years 3.2 Recent years 3.3 Timeline of notable events 3.4 Public holiday 4 Results and records 4.1 Most wins by a horse
4.2 Most wins by a jockey 4.3 Most wins by a trainer 4.4 Most wins by an owner 4.5 Other records 5 Attendance 6 Off the track 7 See also 8 Notes 9 Further
reading 10 External links Qualifying and race conditions ( edit ) 1932 & 1934 winner Peter Pan ridden by jockey Darby Munro , won twice The race is a quality
handicap for horses 3 years old and over , run over a distance of 3,200 metres , on the first Tuesday in November at Flemington Racecourse . The minimum
handicap weight is 50 kg . There is no maximum weight , but the top allocated weight must not be less than 57 kg . The weight allocated to each horse is
declared by the VRC Handicapper in early September .
</item 9>

<item 10>
Melbourne Cup
Maree Lyndon became the first female to ride in the Melbourne Cup , when she partnered Argonaut Style in 1987 , in which she ran second last in the 21 horse
field . In 2004 , Makybe Diva became the first mare to win two cups , and also the first horse to win with different trainers , after David Hall moved to Hong
Kong and transferred her to the Lee Freedman stables . The 2005 Melbourne Cup was held before a crowd of 106,479 . Makybe Diva made history by becoming the only
horse to win the race three times . Trainer Lee Freedman said after the race , `` Go and find the youngest child on the course , because that 's the only person
here who will have a chance of seeing this happen again in their lifetime . '' Due to the 2007 Australian Equine influenza outbreak , believed to have been
started by a horse brought into Australia from Japan , neither Delta Blues nor Pop Rock participated in the 2007 Melbourne Cup . Both horses had been stabled in
Japan .
</item 10>
=== END ITEM id=9f0e159d91d2e019 ===
