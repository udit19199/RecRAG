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

This file has 11 items.

=== ITEM id=c1fdbfeb3a465da7 ===
Gold sentences and facts:
Sentence 0: Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry, famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
  Fact 0.0: Aram Saroyan was born on September 25, 1943.
  Fact 0.1: Aram Saroyan is an American poet.
  Fact 0.2: Aram Saroyan is an American novelist.
  Fact 0.3: Aram Saroyan is an American biographer.
  Fact 0.4: Aram Saroyan is an American memoirist.
  Fact 0.5: Aram Saroyan is an American playwright.
  Fact 0.6: Aram Saroyan is especially known for his minimalist poetry.
  Fact 0.7: A famous example of Aram Saroyan's minimalist poetry is the one-word poem "lighght".
  Fact 0.8: A famous example of Aram Saroyan's minimalist poetry is a one-letter poem comprising a four-legged version of the letter "m".
Sentence 1: Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of microelectronics.
  Fact 1.0: Dr. Edward Keonjian was born on 14 August 1909.
  Fact 1.1: Dr. Edward Keonjian died on 6 September 1999.
  Fact 1.2: Dr. Edward Keonjian was a prominent engineer.
  Fact 1.3: Dr. Edward Keonjian was an early leader in the field of low-power electronics.
  Fact 1.4: Dr. Edward Keonjian was the father of microelectronics.

Retrieved items:
<item 1>
Edward Keonjian
Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of
microelectronics.

 In 1954 Keonjian designed the world's first solar-powered, pocket-sized radio transmitter.

 In 1959 Keonjian designed the first prototype of integrated circuit.

 In 1963 he organized the world's first international symposium on low-power electronics.

 Later on Keonjian collaborated with NASA astronaut Neil Armstrong as chief of failure analysis on the Apollo 11 project.
</item 1>

<item 2>
Edmund Mortimer, 2nd Baron Mortimer
Edmund Mortimer, 2nd Lord Mortimer (1251 – 17 July 1304) was the second son and eventual heir of Roger Mortimer, 1st Baron Mortimer.

 His mother was Maud de Braose.

 As a younger son, Edmund had been intended for clerical or monastic life, and had been sent to study at Oxford University.
</item 2>

<item 3>
Edmund Mortimer, 5th Earl of March
Edmund de Mortimer, 5th Earl of March and 7th Earl of Ulster (6 November 1391 – 18 January 1425) was an English nobleman.

 A great-great-grandson of King Edward III of England, he was heir presumptive to King Richard II of England, his first cousin twice removed, when Richard II
was deposed in favour of Henry IV.

 Edmund Mortimer's claim to the throne was the basis of rebellions and plots against Henry IV and his son Henry V, and was later taken up by the House of York
in the Wars of the Roses, though Mortimer himself was an important and loyal vassal of Henry V and Henry VI.

 Edmund Mortimer was the last Earl of March of the Mortimer family.
</item 3>

<item 4>
Edmund Mortimer (1302–1331)
Sir Edmund Mortimer (1302/1303 – 16 December 1331) was the eldest son of Roger Mortimer, 1st Earl of March and Joan de Geneville, 2nd Baroness Geneville.

 By his wife Elizabeth de Badlesmere he was the father of Roger Mortimer, 2nd Earl of March.

 Though Edmund survived his father by one year, he did not inherit his father's lands and titles as they were forfeited to the Crown and his son only reacquired
them gradually.
</item 4>

<item 5>
Strawberry Saroyan
Strawberry Saroyan (born 1970) is a journalist and author.

 The daughter of Aram Saroyan and granddaughter of playwright William Saroyan and actress Carol Matthau, she spent her childhood in Bolinas, California.

 She has a sister named Cream.

 She writes for the New York Times Style section and is the author of "".
</item 5>

<item 6>
Aram Saroyan
Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry,
famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
</item 6>

<item 7>
My Name Is Aram
My Name is Aram is a book of short stories by William Saroyan first published in 1940.

 The stories detail the exploits of Aram Garoghlanian, a boy of Armenian descent growing up in Fresno, California, and the various members of his large family.

 This book is assigned reading in some schools.
</item 7>

<item 8>
Levan of Imereti
Levan (Georgian: ლევანი ) (1573–1590), of the Bagrationi Dynasty, was a king of Imereti from 1585 to 1588.

 He succeeded on the death of his father, George II, in 1585 when he was twelve years old.

 With his ascend to the throne, Leon faced a revolt by his own uncle, Constantine, who defied the royal authority and took control of Upper Imereti.

 Leon made an alliance with the Mingrelian prince Mamia IV Dadiani, married his sister Marekhi, and forced Constantine to surrender in 1587.

 A year later, Imereti was invaded by Simon I, the resurgent king of Kartli in eastern Georgia, who sought to reunify all Georgian lands under his crown.

 Leon was forced to flee to the highland province of Lechkhumi, but was soon able to resume the throne after Simon had to return to Kartli.
</item 8>

<item 9>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 9>

<item 10>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 10>
=== END ITEM id=c1fdbfeb3a465da7 ===

=== ITEM id=774d139130ee323e ===
Gold sentences and facts:
Sentence 0: In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
  Fact 0.0: In the middle of 1984, a Brazilian company called Prológica brought a new equipment to the Brazilian market
  Fact 0.1: Prológica made its own versions of 8 bits US computers
  Fact 0.2: The new equipment was for Prológica's personal computer series
  Fact 0.3: The new equipment was called "CP"
  Fact 0.4: CP is short for Personal Computer in Portuguese
Sentence 1: The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.
  Fact 1.0: The Atari ST is a line of home computers from Atari Corporation
  Fact 1.1: The Atari ST is the successor to the Atari 8-bit family
Sentence 2:  The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.
  Fact 2.0: The Atari ST is the first personal computer to come with a bitmapped color GUI
  Fact 2.1: The Atari ST's bitmapped color GUI uses a version of Digital Research's GEM
  Fact 2.2: The version of Digital Research's GEM used by the Atari ST was released in February 1985

Retrieved items:
<item 1>
Megamax C
Megamax C is a K&R implementation of C for the Atari ST, Apple IIgs, and Macintosh computers.

 Sold by Megamax, Inc., based in Richardson, Texas, the package includes a one-pass compiler, linker, editor, resource construction kit, and documentation.

 In the early days of the Atari ST, Megamax C was the primary competitor to the Alcyon C compiler from Digital Research which was included in the official
developer kit from Atari.

 According to a review in "Antic" magazine, Megamax C compiled a small benchmark program six times faster than Digital Research's compiler.
</item 1>

<item 2>
DTACK Grounded
DTACK Grounded was a computer hobbyist newsletter published from July 1981 to September 1985 by Hal W. Hardenberg.

 Subtitled "The Journal of Simple 68000 Systems", the newsletter was dedicated to the proposition that the Motorola 68000 CPU could be used to build simple fast
computers.

 In 1981 this was a revolutionary idea.

 This was before 68000-based personal computers like the Apple Macintosh, Commodore Amiga, and Atari ST had been created.

 In 1981 Motorola was marketing the 68000 solely as a CPU for high-end Unix workstations.

 Hal believed that the 68000 could be used as a simple embedded microprocessor as well, and used the newsletter to explain how to do that.
</item 2>

<item 3>
Page 6
Page 6 (subtitled "Atari Users Magazine", and later known as New Atari User) was an independent British publication aimed at users of Atari home computers.

 It was published between 1982 and 1998.

 The computer magazine supported both the Atari 8-bit family of computers (400/800/XL/XE) and the Atari ST range.
</item 3>

<item 4>
Atari Corporation
Atari Corporation was an American manufacturer of computers and video game consoles from 1984 to 1996.

 Atari Corp. was founded in July 1984 when Warner Communications sold the home computing and game console divisions of Atari, Inc. to Jack Tramiel.

 Its chief products were the Atari ST, , Atari 7800, Atari Lynx, and Atari Jaguar.

 The company reverse merged with JTS Inc. in 1996, becoming a small division, which itself closed when JTS liquidated the IP to Hasbro Interactive in 1998.
</item 4>

<item 5>
ATASCII
The ATASCII character set, from ATARI Standard Code for Information Interchange", alternatively ATARI ASCII", is the variation on ASCII used in the Atari 8-bit
family of home computers.

 The first of this family were the Atari 400 and 800, released in 1979, and later models were released throughout the 1980s.

 The last computer to use the ATASCII character set was the Atari XEGS which was released in 1987.

 The Atari ST family of computers used the different Atari ST character set.
</item 5>

<item 6>
XaAES
FreeMint, MultiTOS, and a few more obscure OSes are all successors to TOS the proprietary operating system of the Atari ST computer.

 Later models of the ST Computer were called the TT and the Falcon.

 TOS stands for The Operating System or Tramiel Operating System, as Jack Tramiel was the owner of Atari.

 GEMDOS itself was a near clone of DOS, the IBM PC OS.

 This was most visible to users in the 8.3 character file naming system.

 Like all personal computer operating systems of the day, TOS was single tasking.

 GEM, the graphical user interface, was licensed from Digital Research and was not included in Apple's lawsuit against DR, thus the Macintosh like appearance
and ways of doing things remained on Atari computers.

 As time went on, the big goal for the ST within and without Atari was to have a multitasking TOS.
</item 6>

<item 7>
Happy drives
Happy drives are series of disk drive enhancements for the Atari 8-bit and Atari ST computer families produced by a small company Happy Computers.

 Happy Computers is most noted for the add-in boards for the Atari 810 and Atari 1050 disk drives, which achieved a tremendous speed improvement for reading and
writing, and for the ability to "back up" floppies.

 Happy's products were among the most popular Atari computer add-ons.

 They were still in use and active in the aftermarket as of 2009.
</item 7>

<item 8>
Prológica CP-400
In the middle of 1984 a Brazilian company called Prológica, which made its own versions of 8 bits US computers, brought to the Brazilian market a new equipment
for its personal computer series called "CP" (shorten of Personal Computer in Portuguese).
</item 8>

<item 9>
Atari ST
The Atari ST is a line of home computers from Atari Corporation and the successor to the Atari 8-bit family.

 The initial ST model, the 520ST, saw limited release in April-June 1985 and was widely available in July.

 The Atari ST is the first personal computer to come with a bitmapped color GUI, using a version of Digital Research's GEM released in February 1985.

 The 1040ST, released in 1986, is the first personal computer to ship with a megabyte of RAM in the base configuration and also the first with a
cost-per-kilobyte of less than US$1.
</item 9>

<item 10>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 10>
=== END ITEM id=774d139130ee323e ===

=== ITEM id=5e17b20fa89ff98f ===
Gold sentences and facts:
Sentence 0:  He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.
  Fact 0.0: He is mainly known for his spells at Real Sociedad.
  Fact 0.1: His offensive partnership with Nihat Kahveci at Real Sociedad was one of the best in Spain.
Sentence 1: Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
  Fact 1.0: Nihat Kahveci was born on 23 November 1979.
  Fact 1.1: Nihat Kahveci is a Turkish former footballer.
  Fact 1.2: Nihat Kahveci played as a forward.

Retrieved items:
<item 1>
Lippo Hertzka
Lippo Hertzka (19 November 1904 – 14 March 1951) was a Hungarian football player and manager.

 He played for Essener Turnerbund, MTK Budapest and Real Sociedad.

 After retiring, he coached seven teams, including Real Sociedad and Real Madrid, a team which he coached for 2 years (1930–1932) and led to an undefeated La
Liga championship during the 1931-32 season, which meant the first La Liga title for the "white" squad.

 He also won two league titles in Portugal for Benfica.
</item 1>

<item 2>
Nihat Kahveci
Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
</item 2>

<item 3>
Ivan Toplak
Ivan Toplak (Serbian Cyrillic: Ивaн Toплaк; born September 21, 1931 in Belgrade) is a former Yugoslavian football player and manager.

 He earned 1 cap for Yugoslavia.

 He started his career at Branik in Maribor, where his family moved in 1943 from Belgrade.
</item 3>

<item 4>
C.D. Real Sociedad
Club Deportivo Real Sociedad, commonly known as Real Sociedad (] ), is a Honduran football club based on Tocoa, Colón, Honduras.
</item 4>

<item 5>
2013–14 Real Sociedad season
The 2013–14 season was Real Sociedad's 67th season in La Liga.

 Real Sociedad finished 7th in the league and reached the semifinals of the Copa del Rey.

 The Basque failed to make it out of the group stages of the UEFA Champions League.
</item 5>

<item 6>
Darko Kovačević
Darko Kovačević (] ; Serbian Cyrillic: ; born 18 November 1973) is a Serbian former footballer who played as a forward.

 He began his career in Serbia with Proleter Zrenjanin and subsequently played for Red Star Belgrade, with whom he won a Yugoslav League title and two Yugoslav
Cups.

 His prolific performances earned him a move to Premier League side Sheffield Wednesday, although his time in England was less successful.

 He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.

 Kovačević also had positive spells with Italian club Juventus and Greek side Olympiacos.

 At international level, he represented Yugoslavia at the 1998 FIFA World Cup and at UEFA Euro 2000.
</item 6>

<item 7>
Real Sociedad
Real Sociedad de Fútbol, S.A.D., more commonly referred to as Real Sociedad (] ; "Royal Society") or La Real, is a Spanish football club based in the city of
San Sebastián, Basque Country, founded on 7 September 1909.

 It plays its home matches at the 32,000-capacity Anoeta Stadium.

 Real Sociedad won the Liga title in 1980–81 and 1981–82, and last finished runners-up in 2002–03.

 The club has also won the Copa del Rey twice, in 1909 and 1987.

 It contests the Basque derby against rivals Athletic Bilbao.

 Real Sociedad were founder members of La Liga in 1928, and its longest spell in the top flight was for 40 seasons, from 1967 to 2007.
</item 7>

<item 8>
José Antonio Santamaría Mikel Vaqueriza
José Antonio Santamaría Mikel Vaqueriza (born 16 March 1946 in San Sebastián, Guipúzcoa, Spain; died 19 January 1993) was a Spanish footballer.

 During his career as a professional footballer, Santamaría played for SD Eibar (1963 to 1964), Real Sociedad (1964 to 1967), Real Sociedad (1967 to 1971),
Hercules CF (1971 to 1974) finishing his career at CE Sabadell FC (1974 to 1975).

 After retiring from football he had a career as an entrepreneur in the hospitality sector.

 He was killed in an attack committed by ETA in 1993.
</item 8>

<item 9>
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 9>

<item 10>
Magnus Johansson was a Swedish defender, 1992 Olympian and one-cap international; Ivan Toplak was a Yugoslav player and manager with one cap. Canadian Ostap
Steckiw scored on his sole international appearance and served in the Ukrainian Insurgent Army. Finnish Esko Malm played for and managed Haka and led Finland.
Hungarian Lippo Hertzka coached Real Madrid to its first unbeaten La Liga title and won two Portuguese titles with Benfica. Spanish player José
</item 10>
=== END ITEM id=5e17b20fa89ff98f ===

=== ITEM id=44d1ac21c3ca34a2 ===
Gold sentences and facts:
Sentence 0: Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television host/personality who "TV Guide" called "The Charlie Chaplin of Television".
  Fact 0.0: Alan Young was born Angus Young.
  Fact 0.1: Alan Young was born on November 19, 1919.
  Fact 0.2: Alan Young died on May 19, 2016.
  Fact 0.3: Alan Young was a British-born Canadian-American actor.
  Fact 0.4: Alan Young was a voice actor.
  Fact 0.5: Alan Young was a comedian.
  Fact 0.6: Alan Young was a radio and television host/personality.
  Fact 0.7: TV Guide called Alan Young "The Charlie Chaplin of Television".
Sentence 1:  Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983) and in various other films, TV series and video games until his death.
  Fact 1.0: Alan Young was the voice of Disney's Scrooge McDuck for over thirty years.
  Fact 1.1: Alan Young first voiced Scrooge McDuck in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983).
  Fact 1.2: Alan Young voiced Scrooge McDuck in various other films, TV series and video games.
  Fact 1.3: Alan Young voiced Scrooge McDuck until his death.
Sentence 2:  During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy Awards in 1951.
  Fact 2.0: During the 1940s and 1950s, Alan Young starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television.
  Fact 2.1: The television version of "The Alan Young Show" gained Alan Young two Emmy Awards in 1951.
Sentence 3:  The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.
  Fact 3.0: The cartoon is an adaptation of Charles Dickens' "A Christmas Carol".
  Fact 3.1: The cartoon stars Scrooge McDuck as Ebenezer Scrooge.

Retrieved items:
<item 1>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 1>

<item 2>
A Christmas Carol (2009 film)
A Christmas Carol is a 2009 American 3D computer animated motion-capture fantasy film written and directed by Robert Zemeckis.

 It is an adaptation of the Charles Dickens story of the same name and stars Jim Carrey in a multitude of roles, including Ebenezer Scrooge as a young,
middle-aged, and old man, and the three ghosts who haunt Scrooge.

 The film also features supporting roles done by Gary Oldman, Colin Firth, Bob Hoskins, Robin Wright, and Cary Elwes.
</item 2>

<item 3>
Scrooge (1913 film)
Scrooge is a 1913 British black and white silent film based on the 1843 novel "A Christmas Carol" by Charles Dickens.

 It starred Seymour Hicks as Ebenezer Scrooge.

 In the United States it was released in 1926 as "Old Scrooge".
</item 3>

<item 4>
Scrooge (1935 film)
Scrooge is a 1935 British fantasy film directed by Henry Edwards and starring Seymour Hicks, Donald Calthrop and Robert Cochran.

 Hicks appears as Ebenezer Scrooge, the miser who hates Christmas.

 It was the first sound version of the Charles Dickens classic "A Christmas Carol", not counting a 1928 short subject that now appears to be lost.

 Hicks had previously played the role of Scrooge on the stage many times beginning in 1901, and again in a 1913 British silent film version.
</item 4>

<item 5>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 5>

<item 6>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 6>

<item 7>
Scar (Disney)
Scar is a fictional character who appears in Walt Disney Pictures' 32nd animated feature film "The Lion King" (1994).

 The character is voiced by English actor Jeremy Irons, while his singing voice is provided by both Irons and American actor Jim Cummings, the latter of whom
was hired to replace Irons when the former damaged his singing voice.

 Subsequently, Scar makes minor appearances in the film's sequel " "(1998) and "The Lion King 1½ "(2004), in both of which he is voiced entirely by Cummings, as
well as appearing in the Broadway musical adaptation of the film, in which the role of Scar was originated by American actor John Vickery.
</item 7>

<item 8>
Toy Story 3
The film features an ensemble voice cast with Tom Hanks, Tim Allen, Joan Cusack, Don Rickles, Wallace Shawn, John Ratzenberger, Estelle Harris, Jodi Benson, and
John Morris reprising their roles from the previous films, along with Blake Clark as the voice of Slinky Dog (because Jim Varney died ten years before the
release of the film) and Ned Beatty, Michael Keaton, Whoopi Goldberg, Timothy Dalton, Kristen Schaal, Bonnie Hunt, and Jeff Garlin voicing the new characters
introduced in this film.
</item 8>

<item 9>
Scrooge, or, Marley's Ghost
Scrooge, or, Marley's Ghost is a 1901 British short silent drama film, directed by Walter R. Booth, featuring the miserly Ebenezer Scrooge confronted by
Marley's ghost and given visions of Christmas past, present, and future, is the earliest known film adaptation of Charles Dickens's 1843 novel "A Christmas
Carol".

 The film, "although somewhat flat and stage-bound to modern eyes," according to Ewan Davidson of BFI Screenonline, "was an ambitious undertaking at the time,"
as, "not only did it attempt to tell an 80 page story in five minutes, but it featured impressive trick effects, superimposing Marley's face over the door
knocker and the scenes from his youth over a black curtain in Scrooge's bedroom."
</item 9>

<item 10>
Scrooge (1951 film)
Scrooge is a 1951 British fantasy drama film and an adaptation of Charles Dickens' "A Christmas Carol" (1843).

 It stars Alastair Sim as Ebenezer Scrooge, and was produced and directed by Brian Desmond Hurst, with a screenplay by Noel Langley.

 It was released as A Christmas Carol in the United States.
</item 10>
=== END ITEM id=44d1ac21c3ca34a2 ===

=== ITEM id=08bdc0a28a158cf1 ===
Gold sentences and facts:
Sentence 0: Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and songwriter.
  Fact 0.0: Toshimitsu Deyama was born October 10, 1965 in Tateyama, Chiba.
  Fact 0.1: Toshimitsu Deyama's Japanese name is 出山利三.
  Fact 0.2: Toshimitsu Deyama's name is also written as Deyama Toshimitsu.
  Fact 0.3: Toshimitsu Deyama is known exclusively by his stage name Toshi.
  Fact 0.4: Toshimitsu Deyama is a Japanese singer and songwriter.
Sentence 1:  He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.
  Fact 1.0: Toshimitsu Deyama is best known as the lead vocalist of the heavy metal band X Japan.
  Fact 1.1: Toshimitsu Deyama is a co-founder of X Japan.
  Fact 1.2: X Japan disbanded in 1997.
Sentence 2: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 2.0: Warrel Dane was born March 7, 1969.
  Fact 2.1: Warrel Dane is the lead singer of the American power metal band Sanctuary.

Retrieved items:
<item 1>
Graph query row:
person: Toshimitsu Deyama
occupation: ['singer', 'songwriter']
organization: Sanctuary
is_member: False
role: None
period: None
status: None
</item 1>

<item 2>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 2>

<item 3>
Toshi (musician)
Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and
songwriter.

 He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.

 Toshi then went on to have an extensive solo career.

 In 2007, X Japan reunited and began an ongoing world tour in 2010.

 His distinctive, impassioned tenor vocals have been compared to those of Steve Perry.
</item 3>

<item 4>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 4>

<item 5>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 5>

<item 6>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 6>

<item 7>
Into the Mirror Live
Into The Mirror Live / Black Reflections is live promotional EP of the band Sanctuary recorded on May 12, 1990 at The Country Club in Reseda, California, during
their "Into the Mirror Black" tour.

 Approximately 1000 CD copies of this were made and less than 500 were ever distributed to radio stations and record stores.

 Because of contractual problems with Epic Records, the full version was never released for public consumption.

 Epic Records destroyed 500+ copies of this CD.

 Warrel Dane has stated on several occasions that he has the master tapes and copyrights of the full show and intended to have it finally released.

 To date that has not happened.

 This CD has been heavily pirated over the years as a 2-on-1 CD with the Satan's Host "Metal From Hell" album on the notorious Reborn Classics record label.
</item 7>

<item 8>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 8>

<item 9>
Graph query row:
person: Warrel Dane
occupation: lead singer
organization: Sanctuary
is_member: True
role: None
period: None
status: None
</item 9>

<item 10>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 10>
=== END ITEM id=08bdc0a28a158cf1 ===

=== ITEM id=bc41a82b100ce810 ===
Gold sentences and facts:
Sentence 0:  One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
  Fact 0.0: Updike was one of only three writers to win the Pulitzer Prize for Fiction more than once.
  Fact 0.1: The other two writers who won the Pulitzer Prize for Fiction more than once were Booth Tarkington and William Faulkner.
  Fact 0.2: Updike published more than twenty novels during his career.
  Fact 0.3: Updike published more than a dozen short-story collections during his career.
  Fact 0.4: Updike published poetry during his career.
  Fact 0.5: Updike published art and literary criticism during his career.
  Fact 0.6: Updike published children's books during his career.
Sentence 1:  Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.
  Fact 1.0: Seventeen of Updike's novels were bestsellers.
  Fact 1.1: More than 100 million copies of Updike's books are in print.

Retrieved items:
<item 1>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 1>

<item 2>
Tom Clancy's Net Force Explorers: Gameprey
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 2>

<item 3>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 3>

<item 4>
Tom Clancy's Net Force Explorers: Duel Identity
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 4>

<item 5>
Tom Clancy's Net Force Explorers: Safe House
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 5>

<item 6>
Tom Clancy's Net Force Explorers: High Wire
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 6>

<item 7>
Tom Clancy's Net Force Explorers: Deathworld
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 7>

<item 8>
Tom Clancy's Net Force Explorers: Death Match
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 8>

<item 9>
Tom Clancy's Rainbow 6: Patriots
Tom Clancy's Rainbow 6: Patriots is a cancelled first-person shooter video game, part of the "Tom Clancy's Rainbow Six" series, announced on the cover of the
December 2011 issue of "Game Informer".

 It was to be published by Ubisoft, and was developed by the company's Montreal studio, with additional development by Ubisoft Toronto and Red Storm
Entertainment.

 Due to the death of Tom Clancy in October 2013, concern was raised that this game would become the last to bear his name.

 Ubisoft has since stated that they will continue putting Tom Clancy's name on future Tom Clancy titles out of respect for the late author.
</item 9>

<item 10>
Tom Clancy
Thomas Leo "Tom" Clancy Jr. (April 12, 1947 – October 1, 2013) was an American novelist best known for his technically detailed espionage and military-science
story lines set during and after the Cold War.

 Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.

 His name was also used on movie scripts written by ghost writers, nonfiction books on military subjects, and video games.

 He was a part-owner of the Baltimore Orioles and vice-chairman of their community activities and public affairs committees.
</item 10>
=== END ITEM id=bc41a82b100ce810 ===

=== ITEM id=10ac3b90fa81e7a0 ===
Gold sentences and facts:
Sentence 0:  He earned 1 cap under Jack Charlton for the Irish national team.
  Fact 0.0: The player earned 1 cap for the Irish national team.
  Fact 0.1: The player earned that cap under Jack Charlton.
Sentence 1:  He was part of the England team that won the 1966 World Cup.
  Fact 1.0: The player was part of the England team that won the 1966 World Cup.
  Fact 1.1: The England team won the 1966 World Cup.

Retrieved items:
<item 1>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 1>

<item 2>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 2>

<item 3>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 3>

<item 4>
1929–30 British Home Championship
The 1929-30 British Home Championship was an edition of the annual international football tournament played between the British Home Nations.

 1930 was the year in which the tournament finally gained a serious rival as the premier international football competition, with the inception of the 1930 FIFA
World Cup, held in Uruguay.

 The Home Nations were not however members of FIFA due to disputes over the growing professionalism in continental and South American football.

 As a result, they were not able to attend and indicated that even if they were invited they would have no interest in attending, deeming foreign opposition too
weak for serious contest.

 It is interesting to speculate what would have happened had the Home Nations entered the tournament, especially as the tide of world football was changing
against Britain.

 The England team, which dominated the 1930 championship, had lost to Spain the year before in the first defeat by a foreign football team, and in the same year
they only managed draws with Germany and Austria.

 The Scottish side, which had won most of the previous ten championships, was likewise unprepared, only playing its first game outside the British Isles in
1929, and being heavily defeated on tour in 1931 by both the Austrians and the Italians.
</item 4>

<item 5>
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
</item 5>

<item 6>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 6>

<item 7>
List of Liverpool F.C. players (1–24 appearances)
Liverpool Football Club is an English association football club based in Liverpool, Merseyside.

 The club was formed in 1892 following a disagreement between the board of Everton and club president John Houlding, who owned the club's ground, Anfield.

 The disagreement between the two parties over rent resulted in Everton moving to Goodison Park from Anfield, which left Houlding with an empty stadium.

 Thus, he founded Liverpool F.C. to play in the empty stadium.

 Liverpool won the First Division title for the first time in 1901; since then, the club has won a further 17 league titles, along with seven FA Cups and eight
Football League Cups.

 They have also been crowned champions of European football on five occasions by winning the European Cup/UEFA Champions League in 1977, 1978, 1981, 1984 and
2005.

 The club was one of 22 members of the Premier League when it was formed in 1992.

 They experienced the most successful period in their history under the management of Bob Paisley, who guided the team to 21 trophies in nine seasons.
</item 7>

<item 8>
Barry Rose
Barry Michael Rose OBE, FRAM, FRSCM, Hon.D.Mus, Hon.FGCM, Hon FRCO, M.Univ., (Surrey), born 24 May 1934) is a choir trainer and organist.

 He is best known for founding the choir and the pattern of daily sung Worship at the new Guildford Cathedral in 1961, and later, on 29 July 1981, directing the
music at the wedding of Charles, Prince of Wales and Diana, Princess of Wales formerly Lady Diana Frances Spencer, at St Paul's Cathedral in London.
</item 8>

<item 9>
Barry Murphy (footballer, born 1959)
Barry Murphy (born 1 April 1959 in Dublin) was an Irish soccer player during the 1980s.

 He earned 1 cap under Jack Charlton for the Irish national team.
</item 9>

<item 10>
List of Liverpool F.C. managers
Liverpool Football Club is an English association football club based in Liverpool, Merseyside.

 The club was formed in 1892 following a disagreement between the board of Everton and club president John Houlding, who owned the club's ground, Anfield.

 The disagreement between the two parties over rent resulted in Everton with is the best moving to Goodison Park from Anfield, which left Houlding with no
tenant.

 Thus, he founded Liverpool F.C. to play in the empty stadium.

 Liverpool won the First Division title for the first time in 1901; since then, the club has won a further 17 league titles, along with seven FA Cups and eight
Football League Cups.

 They have also been crowned champions of European football on five occasions by winning the European Cup/UEFA Champions League in 1977, 1978, 1981, 1984 and
2005.

 The club was one of 22 members of the Premier League when it was formed in 1992.
</item 10>
=== END ITEM id=10ac3b90fa81e7a0 ===

=== ITEM id=aad59c204a7e51dd ===
Gold sentences and facts:
Sentence 0: The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.
  Fact 0.0: The presidency of Ronald Reagan began on January 20, 1981
  Fact 0.1: Ronald Reagan was inaugurated as President of the United States on January 20, 1981
  Fact 0.2: The presidency of Ronald Reagan ended on January 20, 1989
Sentence 1: George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice President of the United States from 1981 to 1989.
  Fact 1.0: George Herbert Walker Bush was born June 12, 1924
  Fact 1.1: George Herbert Walker Bush is an American politician
  Fact 1.2: George Herbert Walker Bush was the 41st President of the United States from 1989 to 1993
  Fact 1.3: George Herbert Walker Bush was the 43rd Vice President of the United States from 1981 to 1989

Retrieved items:
<item 1>
Wendell Anderson
Wendell Richard "Wendy" Anderson (February 1, 1933 – July 17, 2016) was an American politician and the 33rd governor of Minnesota, serving from January 4, 1971,
to December 29, 1976.

 In late 1976, he resigned as governor in order to be appointed to the U.S. Senate after Senator Walter Mondale was elected Vice President of the United States.

 Anderson served in the Senate from December 30, 1976, to December 29, 1978 (after losing the 1978 Senate election to Rudy Boschwitz, he resigned a few days
before the end of his term to give Boschwitz seniority).
</item 1>

<item 2>
Second inauguration of Ronald Reagan
The second inauguration of Ronald Reagan as President of the United States was held privately on Sunday, January 20, 1985 in the Grand Foyer of the White House,
and publicly the following day, January 21, 1985 at the United States Capitol in Washington, D.C. The inauguration marked the commencement of the second
four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.
</item 2>

<item 3>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 3>

<item 4>
Presidency of Ronald Reagan
The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.

 Reagan, a Republican, took office as the 40th United States president following a landslide win over Democratic incumbent President Jimmy Carter in the 1980
presidential election.

 Reagan was succeeded by his vice president, George H. W. Bush, who won the 1988 presidential election with Reagan's support.
</item 4>

<item 5>
Elliott Abrams
Elliott Abrams (born January 24, 1948) is a former American diplomat, lawyer and political scientist who served in foreign policy positions for Presidents
Ronald Reagan and George W. Bush.

 Abrams was convicted of withholding information from Congress about the Iran–Contra affair while serving under Reagan, but was pardoned by President George H.
W. Bush.
</item 5>

<item 6>
2012 Libertarian National Convention
The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United
States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.

 Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.

 Retired California state court judge Jim Gray won the vice-presidential nomination, also on the first ballot.

 The convention also chose to replace most of the Libertarian National Committee party officers and members-at-large.
</item 6>

<item 7>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 7>

<item 8>
Reagan Democrat
A Reagan Democrat is a traditionally Democratic voter in the United States, referring especially to white working-class Rust Belt residents, who defected from
their party to support Republican President Ronald Reagan in either or both of the 1980 and 1984 elections as well as Republican Presidents George H. W. Bush in
the 1988 election and George W. Bush in either or both of the 2000 and 2004 elections.

 Part of this group also defected to Donald Trump, the Republican candidate in 2016, who won in Pennsylvania, Ohio, Michigan, Wisconsin, and Iowa, all states
that voted for Reagan but went Democratic in 2008 and 2012.
</item 8>

<item 9>
First inauguration of Ronald Reagan
The first inauguration of Ronald Reagan as the 40th President of the United States was held on Tuesday, January 20, 1981, on the West Front of the United States
Capitol in Washington, D.C..

 This was the first inauguration to be held on the building's west side.

 The inauguration marked the commencement of the first four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.

 Chief Justice Warren E. Burger administered the presidential oath of office to Reagan, who placed his hand upon a family Bible given to him by his mother, open
to .

 Associate Justice Potter Stewart administered the vice presidential oath to Bush.
</item 9>

<item 10>
Ronald Reagan judicial appointment controversies
During President Ronald Reagan's presidency, he nominated at least twelve people for various federal appellate judgeship who were not confirmed.

 In some cases, the nominations were not processed by the Democratic-controlled Senate Judiciary Committee before Reagan's presidency ended, while in other
cases, nominees were rejected by the Senate Judiciary Committee or even blocked by unfriendly members of the Republican Party.

 Three of the nominees were renominated by Reagan's successor, President George H. W. Bush.

 Two of the nominees, Ferdinand Francis Fernandez and Guy G. Hurlbutt, were nominated after July 1, 1988, the traditional start date of the unofficial Thurmond
Rule during a presidential election year.

 Eight of the twelve seats eventually were filled by appointees of President George H. W. Bush.
</item 10>
=== END ITEM id=aad59c204a7e51dd ===

=== ITEM id=53265237eac81cb4 ===
Gold sentences and facts:
Sentence 0: William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story writer.
  Fact 0.0: William Somerset Maugham was born 25 January 1874.
  Fact 0.1: William Somerset Maugham died 16 December 1965.
  Fact 0.2: William Somerset Maugham is better known as W. Somerset Maugham.
  Fact 0.3: William Somerset Maugham was British.
  Fact 0.4: William Somerset Maugham was a playwright.
  Fact 0.5: William Somerset Maugham was a novelist.
  Fact 0.6: William Somerset Maugham was a short story writer.
  Fact 0.7: William Somerset Maugham's name includes the post-nominal CH.
Sentence 1:  He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
  Fact 1.0: W. Somerset Maugham was among the most popular writers of his era.
  Fact 1.1: W. Somerset Maugham was reputedly the highest-paid author during the 1930s.
Sentence 2: Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.
  Fact 2.0: Thomas Merton, O.C.S.O., was born January 31, 1915.
  Fact 2.1: Thomas Merton died December 10, 1968.
  Fact 2.2: Thomas Merton was an American Catholic writer.
  Fact 2.3: Thomas Merton was a theologian.
  Fact 2.4: Thomas Merton was a mystic.
Sentence 3:  In 1949, he was ordained to the priesthood and given the name Father Louis.
  Fact 3.0: In 1949, Thomas Merton was ordained to the priesthood.
  Fact 3.1: Thomas Merton was given the name Father Louis.

Retrieved items:
<item 1>
W. Somerset Maugham
William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story
writer.

 He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
</item 1>

<item 2>
List of works about Thomas Merton
The following is a list of works about Thomas Merton, publications about Thomas Merton, the Trappist monk from Abbey of Gethsemani.

 The bibliography is organized into categories.

 A separate list of works by Thomas Merton is also available.
</item 2>

<item 3>
Thomas Merton
Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.

 A Trappist monk of the Abbey of Gethsemani, Kentucky, he was a poet, social activist, and student of comparative religion.

 In 1949, he was ordained to the priesthood and given the name Father Louis.
</item 3>

<item 4>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 4>

<item 5>
Trio (film)
Trio (also known as W. Somerset Maugham's Trio) is a 1950 British anthology film based on three short stories by W. Somerset Maugham: "The Verger", "Mr.
Know-All" and "Sanatorium".

 Ken Annakin directed "The Verger" and "Mr. Know-All", while Harold French was responsible for "Sanatorium".
</item 5>

<item 6>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 6>

<item 7>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 7>

<item 8>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 8>

<item 9>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 9>

<item 10>
Somerset Maugham Award
The Somerset Maugham Award is a British literary prize given each May by the Society of Authors.

 It is awarded to the best writer or writers of a book published in the past year who is under the age of 35.

 The prize was instituted in 1947 by William Somerset Maugham and thus bears his name.

 The award is to be spent on foreign travel.

 The total fund for each year is £12,000.
</item 10>
=== END ITEM id=53265237eac81cb4 ===

=== ITEM id=584c56d9b3d8fb35 ===
Gold sentences and facts:
Sentence 0:  The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.
  Fact 0.0: The group included Alexander Bogdanov.
  Fact 0.1: The group included Maxim Gorky.
  Fact 0.2: The group included Anatoly Lunacharsky.
  Fact 0.3: The group included Mikhail Pokrovsky.
  Fact 0.4: The group included Grigory Aleksinsky.
  Fact 0.5: The group included Stanislav Volski.
  Fact 0.6: The group included Martyn Liadov.
Sentence 1: Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.
  Fact 1.0: Anatoly Vasilyevich Lunacharsky's Russian name is Анато́лий Васи́льевич Лунача́рский.
  Fact 1.1: Anatoly Vasilyevich Lunacharsky was born on 23 November 1875 (Old Style 11 November 1875).
  Fact 1.2: Anatoly Vasilyevich Lunacharsky died on 26 December 1933.
  Fact 1.3: Anatoly Vasilyevich Lunacharsky was a Russian Marxist revolutionary.
  Fact 1.4: Anatoly Vasilyevich Lunacharsky was the first Soviet People's Commissar of Education.
  Fact 1.5: Anatoly Vasilyevich Lunacharsky was responsible for culture and education as People's Commissar of Education.

Retrieved items:
<item 1>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 1>

<item 2>
You Fell Victim to a Fateful Struggle
You Fell Victim (Russian: "Вы жертвою пали"; "Vy zhertvoiu pali"), or You Fell Victim to a Fateful Struggle, is a Russian Marxist and revolutionary funeral
march.

 It acted as the funeral dirge of the Russian revolutionary movement, among them the Bolsheviks.
</item 2>

<item 3>
Graph query row:
person: Anatoly Lunacharsky
birth_date: 23 November [O.S. 11 November] 1875
occupation: ['Marxist revolutionary', "Soviet People's Commissar of Education", 'art critic', 'journalist']
relationship: MEMBER_OF
related_entity: Vpered
</item 3>

<item 4>
Nikolay Speshnev
Nikolay Alexandrovich Speshnev (Russian: Николай Александрович Спе́шнев; 1821, Kursk - 1882, St. Petersburg) was a 19th-century Russian aristocrat and political
activist, best known for his involvement with the pro-socialist literary discussion group the Petrashevsky Circle.

 He formed a secret revolutionary society from among the members of the circle, which included the young Fyodor Dostoevsky.

 After the government of Tsar Nicholas I arrested the members of the Petrashevsky Circle in 1849, Speshnev was interrogated, threatened with torture, and
eventually sentenced, along with Dostoevsky, Petrashevsky and others, to execution by firing squad.

 The sentence was commuted to hard labour in Siberia, but the prisoners were only informed of this after enduring a mock execution.
</item 4>

<item 5>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 5>

<item 6>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 6>

<item 7>
Petrashevsky Circle
The Petrashevsky Circle was a Russian literary discussion group of progressive-minded commoner-intellectuals in St. Petersburg in the 1840s.

 It was organized by Mikhail Petrashevsky, a follower of the French utopian socialist Charles Fourier.

 Among the members were writers, teachers, students, minor government officials, and army officers.

 While differing in political views, most of them were opponents of the tsarist autocracy and Russian serfdom.

 Like that of the Lyubomudry group founded earlier in the century, the purpose of the circle was to discuss Western philosophy and literature that was
officially banned by the Imperial government of Tsar Nicholas I.
</item 7>

<item 8>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 8>

<item 9>
Vpered
Vpered ("Forward" or "Hasten") (1909 - 1912) was an organization emanating from within the Russian Social Democratic Labour Party (Russian Social Democracy or
RSDLP).

 The faction was gathered by Alexander Bogdanov (1873 - 1928) in December 1909.

 The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.

 Although Vpered emerged from the Bolshevik wing of Russian Social Democracy, the group was critical of Lenin.
</item 9>

<item 10>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 10>
=== END ITEM id=584c56d9b3d8fb35 ===

=== ITEM id=facffa84d1381857 ===
Gold sentences and facts:
Sentence 0: Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.
  Fact 0.0: Pinhead Gunpowder is an American punk rock band.
  Fact 0.1: Pinhead Gunpowder formed in East Bay, California, in 1990.
Sentence 1: Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.
  Fact 1.0: Billie Joe Armstrong was born on February 17, 1972.
  Fact 1.1: Billie Joe Armstrong is an American musician, singer, songwriter and actor.
  Fact 1.2: Billie Joe Armstrong is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day.
  Fact 1.3: Billie Joe Armstrong co-founded Green Day with Mike Dirnt.

Retrieved items:
<item 1>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 1>

<item 2>
Green Day related projects
Since 1991, some members of punk band Green Day have branched out past their "main band" and have started other projects with other musicians and have released
full-length albums and several EPs.

 Notable related projects of Green Day include Billie Joe Armstrong's Pinhead Gunpowder (which also featured Green Day's other guitarist Jason White), The
Frustrators in which Mike Dirnt plays bass, and The Network which many speculate has all three members of Green Day, although under stagenames.
</item 2>

<item 3>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 3>

<item 4>
Graph query row:
person: Billie Joe Armstrong
group: Pinhead Gunpowder
group_kind: punk rock band
role: None
</item 4>

<item 5>
Bill Schneider (musician)
William 'Bill' Schneider (born June 17) is an American musician.

 He has played bass in the punk rock band Pinhead Gunpowder since 1990, as well as providing vocals and writing the lyrics to the song "Backyard Flames".

 He has also been in many other bands such as The Influents, Monsula and Uranium 9v, Sawhorse and The Skinflutes.

 Bill was worked as bass tech during the Nimrod era for Mike Dirnt, Green Day's and as guitar tech on their albums "Nimrod" and "Warning", American Idiot, 21st
Century Breakdown, Uno Dos Tres, Revolution Radio and was their day to day manager from 2003-2012 for their "American Idiot" and 21st Century Breakdown tours
through Uno Dos Tres.

 He also owned "Broken Guitars", a guitar shop in Oakland, California with Billie Joe Armstrong.

 Bill Still works on the sidelines with the band as well as in the studio.
</item 5>

<item 6>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 6>

<item 7>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 7>

<item 8>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 8>

<item 9>
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 9>

<item 10>
Carry the Banner
Carry the Banner is the third EP by the Berkeley, California-based punk rock band Pinhead Gunpowder.

 Originally released on 10" vinyl in December 1994 through Too Many Records, the EP was reissued on CD by Lookout Records shortly after as the initial vinyl
pressing sold out quickly.

 It was the group's first release to feature Jason White on guitar/vocals, replacing Sarah Kirsch, who left the band in 1994 due to differences with Billie Joe
Armstrong after his main band Green Day signed to major label Reprise Records.
</item 10>
=== END ITEM id=facffa84d1381857 ===
