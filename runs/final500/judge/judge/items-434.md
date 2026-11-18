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

=== ITEM id=537225907cc87f47 ===
Gold sentences and facts:
Sentence 0: Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.
  Fact 0.0: Georg Ferdinand Ludwig Philipp Cantor was born on March 3, 1845.
  Fact 0.1: The birth date of Georg Ferdinand Ludwig Philipp Cantor is also given as February 19, 1845 in the Old Style calendar.
  Fact 0.2: Georg Ferdinand Ludwig Philipp Cantor died on January 6, 1918.
  Fact 0.3: Georg Ferdinand Ludwig Philipp Cantor was a German mathematician.
Sentence 1: Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian mathematician.
  Fact 1.0: Vladimir Igorevich Arnold is also spelled Arnol'd.
  Fact 1.1: Vladimir Igorevich Arnold's name in Russian is Влади́мир И́горевич Арно́льд.
  Fact 1.2: Vladimir Igorevich Arnold was born on 12 June 1937.
  Fact 1.3: Vladimir Igorevich Arnold died on 3 June 2010.
  Fact 1.4: Vladimir Igorevich Arnold was a Soviet mathematician.
  Fact 1.5: Vladimir Igorevich Arnold was a Russian mathematician.

Retrieved items:
<item 1>
Cantor space
In mathematics, a Cantor space, named for Georg Cantor, is a topological abstraction of the classical Cantor set: a topological space is a Cantor space if it is
homeomorphic to the Cantor set.

 In set theory, the topological space 2 is called "the" Cantor space.

 Note that, commonly, 2 is referred to simply as the Cantor set, while the term Cantor space is reserved for the more general construction of "D" for a finite
set "D" and a set "S" which might be finite, countable or possibly uncountable.
</item 1>

<item 2>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 2>

<item 3>
Jónsson–Tarski algebra
In mathematics, a Jónsson–Tarski algebra or Cantor algebra is an algebraic structure encoding a bijection from an infinite set "X" onto the product "X"×"X".

 They were introduced by .

 , named them after Georg Cantor because of Cantor's pairing function and Cantor's theorem that an infinite set "X" has the same number of elements as "X"×"X";
the term "Cantor algebra" is also occasionally used to mean the Boolean algebra of all clopen subsets of the Cantor set, or the Boolean algebra of Borel subsets
of the reals modulo meager sets (sometimes called the Cohen algebra).
</item 3>

<item 4>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 4>

<item 5>
Cantor–Bernstein theorem
In set theory and order theory, the Cantor–Bernstein theorem states that the cardinality of the second type class, the class of countable order types, equals
the cardinality of the continuum.

 It was used by Felix Hausdorff and named by him after Georg Cantor and Felix Bernstein.

 Cantor constructed a family of countable order types with the cardinality of the continuum, and in his 1901 inaugural dissertation Bernstein proved that such a
family can have no higher cardinality.
</item 5>

<item 6>
Otto Struve
Otto Struve (August 12, 1897 – April 6, 1963) was a astronomer.

 In Russian, his name is sometimes given as Otto Lyudvigovich Struve (Отто Людвигович Струве); however, he spent most of his life and his entire scientific
career in the United States.

 Otto was the descendant of famous astronomers of the Struve family; he was the son of Ludwig Struve, grandson of Otto Wilhelm von Struve and great-grandson of
Friedrich Georg Wilhelm von Struve.

 He was also the nephew of Karl Hermann Struve.
</item 6>

<item 7>
Georg Cantor
Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.

 He invented set theory, which has become a fundamental theory in mathematics.

 Cantor established the importance of one-to-one correspondence between the members of two sets, defined infinite and well-ordered sets, and proved that the
real numbers are more numerous than the natural numbers.

 In fact, Cantor's method of proof of this theorem implies the existence of an "infinity of infinities".

 He defined the cardinal and ordinal numbers and their arithmetic.

 Cantor's work is of great philosophical interest, a fact of which he was well aware.
</item 7>

<item 8>
Alfred von Arneth
Alfred Ritter von Arneth (10 July 181930 July 1897) was an Austrian historian.

 His principal scholarly work is a ten-volume biography of the Habsburg Empress Maria Theresa, first published in installments from 1863 to 1879 and still
regarded as the standard work on the subject.

 Born at Vienna, he was the son of Joseph Calasanza von Arneth (1791–1863), a well-known historian and archaeologist, who wrote a history of the Austrian Empire
(Vienna, 1827) and several works on numismatics and brother of Doctor Franz Hektor von Arneth (1818–1907).
</item 8>

<item 9>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 9>

<item 10>
Georg Cantor's first set theory article
Georg Cantor's first set theory article was published in 1874 and contains the first theorems of transfinite set theory, which studies infinite sets and their
properties.

 One of these theorems is "Cantor's revolutionary discovery" that the set of all real numbers is uncountably, rather than countably, infinite.

 This theorem is proved using Cantor's first uncountability proof, which differs from the more familiar proof using his diagonal argument.

 The title of the article, "On a Property of the Collection of All Real Algebraic Numbers" ("Ueber eine Eigenschaft des Inbegriffes aller reellen algebraischen
Zahlen "), refers to its first theorem: the set of real algebraic numbers is countable.
</item 10>
=== END ITEM id=537225907cc87f47 ===

=== ITEM id=8b39a4f6f595ad24 ===
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
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
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
</item 5>

<item 6>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 6>

<item 7>
Walter Ward (footballer)
Walter Ward (1869 – after 1890) was an English professional footballer born in Birmingham who played in the Football Alliance for Small Heath.

 During the 1890–91 season, Ward deputised for regular goalkeeper Chris Charsley when Charsley's duties as a serving police officer demanded his absence from
the side.
</item 7>

<item 8>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 8>

<item 9>
Barry Murphy (footballer, born 1959)
Barry Murphy (born 1 April 1959 in Dublin) was an Irish soccer player during the 1980s.

 He earned 1 cap under Jack Charlton for the Irish national team.
</item 9>

<item 10>
Esko Malm
Esko Malm (23 August 1940, Valkeakoski) is a Finnish former footballer and manager who earned 1 cap at international level in 1963.

 At club level Malm played for FC Haka between 1958-72.

 After playing career he became a manager of Haka and in 1979 he was chosen as a manager of Finland.
</item 10>
=== END ITEM id=8b39a4f6f595ad24 ===

=== ITEM id=d375ed410e5bd3a1 ===
Gold sentences and facts:
Sentence 0: Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant", his ten-novel fantasy series.
  Fact 0.0: Stephen Reeder Donaldson was born on May 13, 1947.
  Fact 0.1: Stephen Reeder Donaldson is an American novelist.
  Fact 0.2: Stephen Reeder Donaldson writes fantasy novels.
  Fact 0.3: Stephen Reeder Donaldson writes science fiction novels.
  Fact 0.4: Stephen Reeder Donaldson writes mystery novels.
  Fact 0.5: Stephen Reeder Donaldson is most famous for "The Chronicles of Thomas Covenant".
  Fact 0.6: "The Chronicles of Thomas Covenant" is a ten-novel fantasy series.
  Fact 0.7: "The Chronicles of Thomas Covenant" is a series by Stephen Reeder Donaldson.
Sentence 1: Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.
  Fact 1.0: Michael John Moorcock was born on 18 December 1939.
  Fact 1.1: Michael John Moorcock is an English writer.
  Fact 1.2: Michael John Moorcock is primarily a writer of science fiction.
  Fact 1.3: Michael John Moorcock is primarily a writer of fantasy.
  Fact 1.4: Michael John Moorcock has also published literary novels.

Retrieved items:
<item 1>
Douglas Hill
Douglas Arthur Hill (6 April 1935 – 21 June 2007) was a Canadian science fiction author, editor and reviewer.

 He was born in Brandon, Manitoba, the son of a railroad engineer, and was raised in Prince Albert, Saskatchewan.

 An avid science fiction reader from an early age, he studied English at the University of Saskatchewan (where he earned an Honours B.A. in 1957) and at the
University of Toronto.

 He married fellow writer and U. of S. alumna Gail Robinson (writer) in 1958; they moved to Britain in 1959, where he worked as a freelance writer and editor
for Aldus Books.

 In 1967–1968 he served as Assistant Editor of the controversial "New Worlds" science fiction magazine under Michael Moorcock.
</item 1>

<item 2>
Behold the Man (novel)
Behold the Man (1969) is a science fiction novel by British writer Michael Moorcock.

 It originally appeared as a novella in a 1966 issue of "New Worlds"; later, Moorcock produced an expanded version which was first published in 1969 by Allison
& Busby.

 The title derives from the Gospel of John, Chapter 19, Verse 5: "Then Jesus came out, wearing the crown of thorns and the purple robe.

 And Pilate said to them Behold the Man."
</item 2>

<item 3>
A Cure for Cancer
A Cure for Cancer is a novel by British fantasy and science fiction writer Michael Moorcock, first published in London 1971 by Allison and Busby.

 The book is part of Moorcock's long-running Jerry Cornelius series.
</item 3>

<item 4>
The Chinese Agent
The Chinese Agent (1970) is a comic novel by Michael Moorcock.

 It is a revision of "Somewhere in the Night", which Moorcock published in 1966 under the pseudonym Bill Barclay.

 Although Moorcock is best known as the author of fantasy fiction and science fiction-based parables such as "Behold the Man" and "The Dancers at the End of
Time", here he writes a light-hearted caper that parodies the spy novel genre.
</item 4>

<item 5>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 5>

<item 6>
The Final Programme
The Final Programme is a novel by British science fiction and fantasy writer Michael Moorcock.

 Written in 1965 as the underground culture was beginning to emerge, it was not published for several years.

 Moorcock has stated that publishers at the time considered it was "too freaky".
</item 6>

<item 7>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 7>

<item 8>
Michael Moorcock
Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.

 He is best known for his novels about the character Elric of Melniboné, a seminal influence on the field of fantasy in the 1960s and 1970s.
</item 8>

<item 9>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 9>

<item 10>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 10>
=== END ITEM id=d375ed410e5bd3a1 ===

=== ITEM id=1d82a166e54786a4 ===
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
Scrooge (1951 film)
Scrooge is a 1951 British fantasy drama film and an adaptation of Charles Dickens' "A Christmas Carol" (1843).

 It stars Alastair Sim as Ebenezer Scrooge, and was produced and directed by Brian Desmond Hurst, with a screenplay by Noel Langley.

 It was released as A Christmas Carol in the United States.
</item 1>

<item 2>
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
</item 2>

<item 3>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 3>

<item 4>
Scar (Disney)
Scar is a fictional character who appears in Walt Disney Pictures' 32nd animated feature film "The Lion King" (1994).

 The character is voiced by English actor Jeremy Irons, while his singing voice is provided by both Irons and American actor Jim Cummings, the latter of whom
was hired to replace Irons when the former damaged his singing voice.

 Subsequently, Scar makes minor appearances in the film's sequel " "(1998) and "The Lion King 1½ "(2004), in both of which he is voiced entirely by Cummings, as
well as appearing in the Broadway musical adaptation of the film, in which the role of Scar was originated by American actor John Vickery.
</item 4>

<item 5>
Scrooge (1913 film)
Scrooge is a 1913 British black and white silent film based on the 1843 novel "A Christmas Carol" by Charles Dickens.

 It starred Seymour Hicks as Ebenezer Scrooge.

 In the United States it was released in 1926 as "Old Scrooge".
</item 5>

<item 6>
Scrooge, or, Marley's Ghost
Scrooge, or, Marley's Ghost is a 1901 British short silent drama film, directed by Walter R. Booth, featuring the miserly Ebenezer Scrooge confronted by
Marley's ghost and given visions of Christmas past, present, and future, is the earliest known film adaptation of Charles Dickens's 1843 novel "A Christmas
Carol".

 The film, "although somewhat flat and stage-bound to modern eyes," according to Ewan Davidson of BFI Screenonline, "was an ambitious undertaking at the time,"
as, "not only did it attempt to tell an 80 page story in five minutes, but it featured impressive trick effects, superimposing Marley's face over the door
knocker and the scenes from his youth over a black curtain in Scrooge's bedroom."
</item 6>

<item 7>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 7>

<item 8>
Toy Story 3
The film features an ensemble voice cast with Tom Hanks, Tim Allen, Joan Cusack, Don Rickles, Wallace Shawn, John Ratzenberger, Estelle Harris, Jodi Benson, and
John Morris reprising their roles from the previous films, along with Blake Clark as the voice of Slinky Dog (because Jim Varney died ten years before the
release of the film) and Ned Beatty, Michael Keaton, Whoopi Goldberg, Timothy Dalton, Kristen Schaal, Bonnie Hunt, and Jeff Garlin voicing the new characters
introduced in this film.
</item 8>

<item 9>
Scrooge (1935 film)
Scrooge is a 1935 British fantasy film directed by Henry Edwards and starring Seymour Hicks, Donald Calthrop and Robert Cochran.

 Hicks appears as Ebenezer Scrooge, the miser who hates Christmas.

 It was the first sound version of the Charles Dickens classic "A Christmas Carol", not counting a 1928 short subject that now appears to be lost.

 Hicks had previously played the role of Scrooge on the stage many times beginning in 1901, and again in a 1913 British silent film version.
</item 9>

<item 10>
A Christmas Carol (2009 film)
A Christmas Carol is a 2009 American 3D computer animated motion-capture fantasy film written and directed by Robert Zemeckis.

 It is an adaptation of the Charles Dickens story of the same name and stars Jim Carrey in a multitude of roles, including Ebenezer Scrooge as a young,
middle-aged, and old man, and the three ghosts who haunt Scrooge.

 The film also features supporting roles done by Gary Oldman, Colin Firth, Bob Hoskins, Robin Wright, and Cary Elwes.
</item 10>
=== END ITEM id=1d82a166e54786a4 ===

=== ITEM id=5f9dcfda77dc7cd2 ===
Gold sentences and facts:
Sentence 0:  His primary jockey is Tony McCoy.
  Fact 0.0: The unnamed subject of this sentence has Tony McCoy as its primary jockey.
  Fact 0.1: Tony McCoy is a primary jockey.
Sentence 1:  Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.
  Fact 1.0: Tony McCoy is based in Ireland and the UK.
  Fact 1.1: Tony McCoy rode a record 4,358 winners.
  Fact 1.2: Tony McCoy was Champion Jockey a record 20 consecutive times.
  Fact 1.3: Tony McCoy was Champion Jockey every year he was a professional.

Retrieved items:
<item 1>
German flat racing Champion Jockey
The Champion Jockey of flat racing in Germany is the jockey who has ridden the most winning horses during a season.

 The list of the Champion Jockey started 1870.

 The list below shows the Champion Jockey for each year since 1984.
</item 1>

<item 2>
Binocular (horse)
Binocular, foaled on March 17, 2004 in France, is a French thoroughbred retired racehorse, born to sire, Enrique and dam Bleu Ciel Et Blanc.

 He is owned by J. P. McManus and trained by Nicky Henderson.

 His primary jockey is Tony McCoy.
</item 2>

<item 3>
Ernest Piggott
Ernest "Ernie" Piggott (1878–1967) born Nantwich, Cheshire, England was a leading British jump racing jockey, whose family has become one of the leading
dynasties in British horseracing.

 He was three times Champion Jockey and three times Grand National winner.

 His son, (Ernest) Keith Piggott (1904–1993), was also a leading jump jockey and National-winning trainer, while his grandson is the 11-times British flat
racing Champion Jockey, Lester Piggott.
</item 3>

<item 4>
British flat racing Champion Jockey
The Champion Jockey of flat racing in Great Britain is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey and the number of winners for each year since 1840.

 The seasonal record of jockeys' winners was published for the first time in 1846.

 The championship was sponsored for the first time in 2009 by online casino 32Red and is currently sponsored by Stobart Group.

 The 2016 title was determined by the number of winners ridden between 30 April and 15 October, in both turf and all-weather racing.

 Prior to 2015 the title was decided over the full length of the Flat turf season, from March to November, and carried no prize money to the winner.

 A prize of £25,000 to the champion jockey was also introduced as part of the 2015 changes.
</item 4>

<item 5>
Irish jump racing Champion Jockey
The Champion Jockey of National Hunt racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1947.

 The current champion, as of 2017, is Ruby Walsh.
</item 5>

<item 6>
Fred Winter
Frederick Thomas Winter CBE (20 September 1926 – 5 April 2004) was a British National Hunt racing racehorse jockey and trainer.

 He was British jump racing Champion Jockey four times and British jump racing Champion Trainer eight times.

 He is the only person to have won the Cheltenham Gold Cup, Champion Hurdle and Grand National as both jockey and trainer.

 Winter won the Grand National four times, as a jockey in 1957 (Sundew) and 1962 (Kilmore), and as a trainer in 1965 (Jay Trump) and 1966 (Anglo).
</item 6>

<item 7>
Tony McCoy
Sir Anthony Peter McCoy, OBE (born 4 May 1974), commonly known as A. P. McCoy or Tony McCoy, is a Northern Irish former horse racing jockey.

 Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.

 He stands 1.78 m (5'10"), far taller than most jockeys.
</item 7>

<item 8>
Gordon Richards
Sir Gordon Richards (5 May 1904 – 10 November 1986) was an English jockey.

 He was the British flat racing Champion Jockey 26 times and is often considered the world's greatest ever jockey.

 He remains the only flat jockey to have been knighted.
</item 8>

<item 9>
Irish flat racing Champion Jockey
The Champion Jockey of flat racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1950.
</item 9>

<item 10>
Bart Cummings
James Bartholomew Cummings {'1': ", '2': ", '3': ", '4': "} (14 November 1927 – 30 August 2015), also known by his initials J. B. Cummings, was one of the most
successful Australian racehorse trainers.

 He was known as the "Cups King", referring to the Melbourne Cup, as he won 'the race that stops a nation' a record twelve times.

 During his lifetime Cummings was considered an Australian cultural icon and an Australian National Living Treasure.
</item 10>
=== END ITEM id=5f9dcfda77dc7cd2 ===

=== ITEM id=96209110e8bfbda8 ===
Gold sentences and facts:
Sentence 0: Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.
  Fact 0.0: Lee Dong-hae was born on October 15, 1986.
  Fact 0.1: Lee Dong-hae's name in Hangul is 이동해.
  Fact 0.2: Lee Dong-hae's name in Hanja is 李東海.
  Fact 0.3: Lee Dong-hae is a South Korean singer.
  Fact 0.4: Lee Dong-hae is a South Korean songwriter.
  Fact 0.5: Lee Dong-hae is a South Korean actor.
Sentence 1: Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.
  Fact 1.0: Costadinos Contostavlos was born on 11 June 1987.
  Fact 1.1: Costadinos Contostavlos is better known by his stage name Dappy.
  Fact 1.2: Costadinos Contostavlos is an English singer.
  Fact 1.3: Costadinos Contostavlos is an English songwriter.
  Fact 1.4: Costadinos Contostavlos is an English rapper.
  Fact 1.5: Costadinos Contostavlos is an English actor.

Retrieved items:
<item 1>
Graph query row:
person: Dappy
nationality: English
</item 1>

<item 2>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 2>

<item 3>
Lee Won-il
Lee Won-il (; born December 12, 1979) is a South Korean chef and television personality.

 He is currently a cast member in the variety show "Please Take Care of My Refrigerator".

 He is owner and chef of "Dear.

 bread".
</item 3>

<item 4>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 4>

<item 5>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 5>

<item 6>
Eunhyuk
Lee Hyuk-jae (born April 4, 1986), better known by his stage name Eunhyuk, is a South Korean singer-songwriter and actor.

 He is a member of the South Korean boy group Super Junior and their subgroups, Super Junior-T and Super Junior-H.

 In 2011, he joined Super Junior's Mandopop subgroup, Super Junior-M and the duo Donghae & Eunhyuk; active in China and Japan respectively.
</item 6>

<item 7>
Graph query row:
person: Lee Dong-hae
nationality: South Korean
</item 7>

<item 8>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 8>

<item 9>
Dappy
Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.

 He is best known for being the lead singer of the Camden-based grime trio N-Dubz, with his cousin Tulisa, and Fazer.

 He was known for his slightly unusual dress sense and his love for what he describes as "eye-catching headwear".

 His trademark was wearing a selection of woolly chullos (sometimes referred to as "Dappy hats"), turning up one or both of the ear-flaps of the hat.
</item 9>

<item 10>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 10>
=== END ITEM id=96209110e8bfbda8 ===

=== ITEM id=eaf782a72268976a ===
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
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 1>

<item 2>
Nikolay Speshnev
Nikolay Alexandrovich Speshnev (Russian: Николай Александрович Спе́шнев; 1821, Kursk - 1882, St. Petersburg) was a 19th-century Russian aristocrat and political
activist, best known for his involvement with the pro-socialist literary discussion group the Petrashevsky Circle.

 He formed a secret revolutionary society from among the members of the circle, which included the young Fyodor Dostoevsky.

 After the government of Tsar Nicholas I arrested the members of the Petrashevsky Circle in 1849, Speshnev was interrogated, threatened with torture, and
eventually sentenced, along with Dostoevsky, Petrashevsky and others, to execution by firing squad.

 The sentence was commuted to hard labour in Siberia, but the prisoners were only informed of this after enduring a mock execution.
</item 2>

<item 3>
Petrashevsky Circle
The Petrashevsky Circle was a Russian literary discussion group of progressive-minded commoner-intellectuals in St. Petersburg in the 1840s.

 It was organized by Mikhail Petrashevsky, a follower of the French utopian socialist Charles Fourier.

 Among the members were writers, teachers, students, minor government officials, and army officers.

 While differing in political views, most of them were opponents of the tsarist autocracy and Russian serfdom.

 Like that of the Lyubomudry group founded earlier in the century, the purpose of the circle was to discuss Western philosophy and literature that was
officially banned by the Imperial government of Tsar Nicholas I.
</item 3>

<item 4>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 4>

<item 5>
You Fell Victim to a Fateful Struggle
You Fell Victim (Russian: "Вы жертвою пали"; "Vy zhertvoiu pali"), or You Fell Victim to a Fateful Struggle, is a Russian Marxist and revolutionary funeral
march.

 It acted as the funeral dirge of the Russian revolutionary movement, among them the Bolsheviks.
</item 5>

<item 6>
Apollinariya Yakubova
Apollinariya Yakubova (died 1913 or 1917) was a Russian Marxist revolutionary and, with Vladimir Lenin, one of the founders of the League of Struggle for the
Emancipation of the Working Class.
</item 6>

<item 7>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 7>

<item 8>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 8>

<item 9>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 9>

<item 10>
Vpered
Vpered ("Forward" or "Hasten") (1909 - 1912) was an organization emanating from within the Russian Social Democratic Labour Party (Russian Social Democracy or
RSDLP).

 The faction was gathered by Alexander Bogdanov (1873 - 1928) in December 1909.

 The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.

 Although Vpered emerged from the Bolshevik wing of Russian Social Democracy, the group was critical of Lenin.
</item 10>
=== END ITEM id=eaf782a72268976a ===

=== ITEM id=d3ac411a2ce18f47 ===
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
Tom Clancy's Net Force Explorers: Duel Identity
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 1>

<item 2>
Tom Clancy's Net Force Explorers: Gameprey
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 2>

<item 3>
Tom Clancy's Rainbow 6: Patriots
Tom Clancy's Rainbow 6: Patriots is a cancelled first-person shooter video game, part of the "Tom Clancy's Rainbow Six" series, announced on the cover of the
December 2011 issue of "Game Informer".

 It was to be published by Ubisoft, and was developed by the company's Montreal studio, with additional development by Ubisoft Toronto and Red Storm
Entertainment.

 Due to the death of Tom Clancy in October 2013, concern was raised that this game would become the last to bear his name.

 Ubisoft has since stated that they will continue putting Tom Clancy's name on future Tom Clancy titles out of respect for the late author.
</item 3>

<item 4>
Tom Clancy's Net Force Explorers: High Wire
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 4>

<item 5>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 5>

<item 6>
Tom Clancy's Net Force Explorers: Deathworld
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 6>

<item 7>
Joe McGinniss
Joseph R. McGinniss, Sr. (December 9, 1942 – March 10, 2014), known as Joe McGinniss, was an American non-fiction writer and novelist.

 The author of twelve books, he first came to prominence with the best-selling "The Selling of the President 1968" which described the marketing of
then-presidential candidate Richard Nixon.

 He is popularly known for his trilogy of bestselling true crime books — "Fatal Vision", "Blind Faith" and "Cruel Doubt" — which were adapted into TV miniseries
in the 1980s and 90s.

 His last book was "The Rogue: Searching for the Real Sarah Palin", an account of Sarah Palin, the former governor of Alaska who was the 2008 Republican
vice-presidential nominee.
</item 7>

<item 8>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 8>

<item 9>
Tom Clancy
Thomas Leo "Tom" Clancy Jr. (April 12, 1947 – October 1, 2013) was an American novelist best known for his technically detailed espionage and military-science
story lines set during and after the Cold War.

 Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.

 His name was also used on movie scripts written by ghost writers, nonfiction books on military subjects, and video games.

 He was a part-owner of the Baltimore Orioles and vice-chairman of their community activities and public affairs committees.
</item 9>

<item 10>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 10>
=== END ITEM id=d3ac411a2ce18f47 ===

=== ITEM id=ae2f04d6895013b6 ===
Gold sentences and facts:
Sentence 0:  He earned 1 cap under Jack Charlton for the Irish national team.
  Fact 0.0: The player earned 1 cap for the Irish national team.
  Fact 0.1: The player earned that cap under Jack Charlton.
Sentence 1:  He was part of the England team that won the 1966 World Cup.
  Fact 1.0: The player was part of the England team that won the 1966 World Cup.
  Fact 1.1: The England team won the 1966 World Cup.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 2>

<item 3>
Barry Murphy (footballer, born 1959)
Barry Murphy (born 1 April 1959 in Dublin) was an Irish soccer player during the 1980s.

 He earned 1 cap under Jack Charlton for the Irish national team.
</item 3>

<item 4>
Dave Bailey (athlete)
He was a member of 9 Canadian National Track and Field Teams competing at World University Games, Budapest 1965 and Tokyo 1967, Commonwealth Games, Kingston,
Jamaica, 1966, Pan American Games, Winnipeg, 1967, Commonwealth vs USA, Los Angeles, 1967, Pre-Olympic Games, Mexico City, 1967, Soviet Union Tour, 1968,
European Tour, 1968, Olympic Games, Mexico City, 1968.
</item 4>

<item 5>
Arthur Milton
Clement Arthur Milton (10 March 1928 – 25 April 2007) was an English cricketer and footballer.

 He played County cricket for Gloucestershire from 1948 to 1974, playing six Test matches for England in 1958 and 1959.

 He also played domestic football for Arsenal between 1951 and 1955, and then for a brief period for Bristol City.

 He played one match for England in 1951, against Austria at Wembley.

 He was the last man, and the last survivor, of the twelve people to have played at the highest international level for both England's football and cricket
teams.
</item 5>

<item 6>
John Murphy (fiddle player)
John (Boss) Murphy (1875–1955) from The Leap, Churchtown, Co.

 Cork was a former farmer by profession, but possessed a keen interest in fiddle playing, and was renowned locally for his ability on the instrument.

 His father, William Murphy (1829–1911), was both a fiddle player and maker, indeed John was the proud owner of an instrument that had been made by his father.

 John’s father was his first fiddle teacher.

 His siblings, a brother and three sisters, also played the instrument but their interest in music waned as they reached adulthood and they did not continue to
play.

 John’s musical literacy was also gained primarily from his father, who had learnt to read music at a hedge school at Ballygrace in the locality taught by a
Thomas Croke.

 The same Thomas Croke, many years later, stayed for long periods of time in the Murphy household and would undoubtedly also have taught the young John
directly.
</item 6>

<item 7>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 7>

<item 8>
Walter Ward (footballer)
Walter Ward (1869 – after 1890) was an English professional footballer born in Birmingham who played in the Football Alliance for Small Heath.

 During the 1890–91 season, Ward deputised for regular goalkeeper Chris Charsley when Charsley's duties as a serving police officer demanded his absence from
the side.
</item 8>

<item 9>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 9>

<item 10>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 10>
=== END ITEM id=ae2f04d6895013b6 ===

=== ITEM id=4d9b924f97a1f3c8 ===
Gold sentences and facts:
Sentence 0: Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.
  Fact 0.0: Pagliacci is an Italian opera.
  Fact 0.1: Pagliacci has a prologue and two acts.
  Fact 0.2: Pagliacci has music and libretto by Ruggero Leoncavallo.
  Fact 0.3: The literal translation of Pagliacci is 'Clowns'.
Sentence 1: Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.
  Fact 1.0: Suor Angelica (Sister Angelica) is an opera.
  Fact 1.1: Suor Angelica is in one act.
  Fact 1.2: Suor Angelica is by Giacomo Puccini.
  Fact 1.3: Suor Angelica has an original Italian libretto by Giovacchino Forzano.

Retrieved items:
<item 1>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 1>

<item 2>
Giovacchino Forzano
Giovacchino Forzano (] ; 19 November 1884 – 28 October 1970) was an Italian playwright, librettist, stage director, and film director.

 A resourceful writer, he authored numerous popular plays and produced opera librettos for most of the major Italian composers of the early twentieth century,
including the librettos for Giacomo Puccini's "Suor Angelica" and "Gianni Schicchi".
</item 2>

<item 3>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 3>

<item 4>
Stiffelio
Stiffelio is an opera in three acts by Giuseppe Verdi, from an Italian libretto by Francesco Maria Piave.

 The origin of this was the novel “Le pasteur d’hommes”, by Émile Souvestre, which was published in 1838.

 This was adapted into the French play "Le pasteur, ou L'évangile et le foyer" by Émile Souvestre and Eugène Bourgeois.

 That was in turn translated into Italian by Gaetano Vestri as "Stifellius"; this formed the basis of Piave's libretto.
</item 4>

<item 5>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 5>

<item 6>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 6>

<item 7>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 7>

<item 8>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 8>

<item 9>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 9>

<item 10>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 10>
=== END ITEM id=4d9b924f97a1f3c8 ===

=== ITEM id=65154797e32b5e81 ===
Gold sentences and facts:
Sentence 0: Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.
  Fact 0.0: Robin R. Bottin was born on April 1, 1959.
  Fact 0.1: Robin R. Bottin is an American special make-up effects creator.
Sentence 1:  Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing", with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".
  Fact 1.0: Robin R. Bottin is known for collaborations with directors John Carpenter, Paul Verhoeven and David Fincher.
  Fact 1.1: Robin R. Bottin worked with John Carpenter on “The Fog”.
  Fact 1.2: Robin R. Bottin worked with John Carpenter on “The Thing”.
  Fact 1.3: Robin R. Bottin worked with Paul Verhoeven on “RoboCop”.
  Fact 1.4: Robin R. Bottin worked with Paul Verhoeven on “Total Recall”.
  Fact 1.5: Robin R. Bottin worked with Paul Verhoeven on “Basic Instinct”.
  Fact 1.6: Robin R. Bottin worked with David Fincher on “Se7en”.
  Fact 1.7: Robin R. Bottin worked with David Fincher on “Fight Club”.
Sentence 2: David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.
  Fact 2.0: David Andrew Leo Fincher was born on August 28, 1962.
  Fact 2.1: David Andrew Leo Fincher is an American director and producer.
  Fact 2.2: David Andrew Leo Fincher is notably known for films, television series and music videos.
Sentence 3:  He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).
  Fact 3.0: David Fincher was nominated for the Academy Award for Best Director for “The Curious Case of Benjamin Button” (2008).
  Fact 3.1: David Fincher was nominated for the Academy Award for Best Director for “The Social Network” (2010).
Sentence 4:  For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
  Fact 4.0: For “The Social Network”, David Fincher won the Golden Globe Award for Best Director.
  Fact 4.1: For “The Social Network”, David Fincher won the BAFTA Award for Best Direction.

Retrieved items:
<item 1>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 1>

<item 2>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 2>

<item 3>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 3>

<item 4>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 4>

<item 5>
Meet the Robinsons
Meet the Robinsons is a 2007 American computer-animated science fiction comedy film produced by Walt Disney Animation Studios and released by Walt Disney
Pictures on March 30, 2007.

 The 47th Disney animated feature film, it was released in standard and Disney Digital 3-D versions.

 The film is loosely based on characters from the children's book "A Day with Wilbur Robinson", by William Joyce.

 The voice cast includes Jordan Fry, Wesley Singerman, Harland Williams, Tom Kenny, Steve Anderson, Laurie Metcalf, Adam West, Tom Selleck, and Angela Bassett.

 It was the first film released after John Lasseter became chief creative officer of Walt Disney Animation Studios.
</item 5>

<item 6>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 6>

<item 7>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 7>

<item 8>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 8>

<item 9>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 9>

<item 10>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 10>
=== END ITEM id=65154797e32b5e81 ===
