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

This file has 12 items.

=== ITEM id=63e78ddebeb1cc20 ===
Gold sentences and facts:
Sentence 0: Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.
  Fact 0.0: Scott Richard Weiland was born Scott Richard Kline.
  Fact 0.1: Scott Richard Weiland was born on October 27, 1967.
  Fact 0.2: Scott Richard Weiland's date range ends on December 3, 2015.
  Fact 0.3: Scott Richard Weiland was an American musician.
  Fact 0.4: Scott Richard Weiland was an American singer.
  Fact 0.5: Scott Richard Weiland was an American songwriter.
Sentence 1: Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.
  Fact 1.0: Richard Willis Hawley was born on 17 January 1967.
  Fact 1.1: Richard Willis Hawley is English.
  Fact 1.2: Richard Willis Hawley is a guitarist.
  Fact 1.3: Richard Willis Hawley is a singer-songwriter.
  Fact 1.4: Richard Willis Hawley is a producer.

Retrieved items:
<item 1>
Scott Weiland and the Wildabouts
Scott Weiland and the Wildabouts (sometimes billed simply as The Wildabouts) were a rock and roll band formed sometime around 2012 by former Stone Temple Pilots
and Velvet Revolver lead singer Scott Weiland, along with bassist Tommy Baker, guitarist Jeremy Brown, and multi-instrumentalist Doug Grean.

 The Wildabouts were marred with several roster changes (including the death of guitarist Jeremy Brown) and ultimately disbanded after Weiland's sudden death on
a tour bus in late 2015.
</item 1>

<item 2>
Standing at the Sky's Edge
Standing at the Sky's Edge is the seventh studio album from English musician Richard Hawley, released in the UK on 7 May 2012 and in the US on 28 August 2012.

 The album is markedly different from Hawley's previous efforts, often relinquishing softer instrumentation in favour of squalling guitars.

 As with all of Hawley's previous albums, the title obliquely refers to a location in his native city of Sheffield, in this case Skye Edge, a hillside area with
views over the city centre and formerly known for its crime-ridden estates but largely redeveloped in the mid-2000s.
</item 2>

<item 3>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 3>

<item 4>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 4>

<item 5>
A Compilation of Scott Weiland Cover Songs
A Compilation of Scott Weiland Cover Songs is a compilation album and third solo studio album by Stone Temple Pilots frontman Scott Weiland, released on August
30, 2011.

 The album consists entirely of cover songs, featuring an array of covers of artists that inspired Weiland, such as David Bowie, The Beatles, The Rolling
Stones, and The Smiths.

 The album was originally set to be released together with Weiland's memoir "Not Dead & Not for Sale" but Weiland decided later to release the album by itself
exclusively in digital format.
</item 5>

<item 6>
Hugh Whitaker
Hugh Whitaker (born 18 May 1961) is an English musician and the former drummer for the British indie rock band The Housemartins.

 He replaced original drummer Chris Lang and drummed for the band's first album, "London 0 Hull 4", and its attendant single releases.

 He left the band before the recording of their second album, "The People Who Grinned Themselves to Death".

 Whitaker left the band on amicable terms and even participated in the promotional video for the band's first single without him, "Five Get Over Excited",
wherein he was kidnapped by his replacement, Dave Hemingway, and locked in a hessian sack.
</item 6>

<item 7>
John Scott Whiteley
John Scott Whiteley (born 1950) is an English organist and composer.

 He has performed extensively around the world and since 1985 has undertaken an annual tour of the USA.

 He has performed in most major UK Cathedrals and concert halls, and was Assistant Organist and later Organist and Director of the Girls' choir at York Minster
between 1975 and 2010.

 He is currently Organist Emeritus of York Minster.
</item 7>

<item 8>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 8>

<item 9>
Richard Hawley
Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.

 After his first band Treebound Story (formed while he was still at school) broke up, Hawley found success as a member of Britpop band Longpigs in the 1990s.

 After that group broke up in 2000, he later joined the band Pulp, led by his friend Jarvis Cocker, for a short time.

 As a solo musician, Hawley has released seven studio albums.

 He has been nominated for a Mercury prize twice and once for a Brit Award.

 He has collaborated with Lisa Marie Presley, Arctic Monkeys, Manic Street Preachers and Paul Weller.
</item 9>

<item 10>
Bill Scott (author)
William Neville "Bill" Scott OAM (4 October 1923 – 22 December 2005) was an Australian author, folklorist, songwriter, poet and a collector of bush ballads and
Australian folk history.

 He has published anthologies of Australian bush songs, including the best selling book "The Complete Book of Australian Folklore" published in 1976.

 He was awarded the Order of Australia in 1992 for his contributions to folklore, folk music and Australian literature.

 He was considered a living treasure and his anthologies of songs and his donated collections continue his legacy.
</item 10>
=== END ITEM id=63e78ddebeb1cc20 ===

=== ITEM id=d3792a7f5589499c ===
Gold sentences and facts:
Sentence 0: Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.
  Fact 0.0: Claudio Giovanni Antonio Monteverdi was baptized on 15 May 1567.
  Fact 0.1: Claudio Giovanni Antonio Monteverdi died on 29 November 1643.
  Fact 0.2: Claudio Giovanni Antonio Monteverdi was an Italian composer.
  Fact 0.3: Claudio Giovanni Antonio Monteverdi was an Italian string player.
  Fact 0.4: Claudio Giovanni Antonio Monteverdi was an Italian choirmaster.
Sentence 1:  A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
  Fact 1.0: Monteverdi composed both secular and sacred music.
  Fact 1.1: Monteverdi was a pioneer in the development of opera.
  Fact 1.2: Monteverdi is considered a crucial transitional figure between the Renaissance and the Baroque periods of music history.
Sentence 2: Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.
  Fact 2.0: Georges Bizet was born on 25 October 1838.
  Fact 2.1: Georges Bizet died on 3 June 1875.
  Fact 2.2: Georges Bizet was registered at birth as Alexandre César Léopold Bizet.
  Fact 2.3: Georges Bizet was a French composer.
  Fact 2.4: Georges Bizet was a composer of the romantic era.
Sentence 3:  Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the most popular and frequently performed works in the entire opera repertoire.
  Fact 3.0: Georges Bizet is best known for his operas.
  Fact 3.1: Bizet's career was cut short by his early death.
  Fact 3.2: Bizet achieved few successes before his final work.
  Fact 3.3: Bizet's final work is 'Carmen'.
  Fact 3.4: Carmen has become one of the most popular works in the entire opera repertoire.
  Fact 3.5: Carmen is one of the most frequently performed works in the entire opera repertoire.

Retrieved items:
<item 1>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 1>

<item 2>
Georges Bizet
Georges Bizet (] ; 25 October 18383 June 1875), registered at birth as Alexandre César Léopold Bizet, was a French composer of the romantic era.

 Best known for his operas in a career cut short by his early death, Bizet achieved few successes before his final work, "Carmen", which has become one of the
most popular and frequently performed works in the entire opera repertoire.
</item 2>

<item 3>
Claudio Monteverdi
Claudio Giovanni Antonio Monteverdi (] ; 15 May 1567 (baptized) – 29 November 1643) was an Italian composer, string player and choirmaster.

 A composer of both secular and sacred music, and a pioneer in the development of opera, he is considered a crucial transitional figure between the Renaissance
and the Baroque periods of music history.
</item 3>

<item 4>
John Whenham
John Whenham is an English musicologist and academic who specializes in early Italian baroque music.

 He earned both a Bachelor of Music and a Master of Music from the University of Nottingham, and a Doctor of Philosophy from the University of Oxford.

 He is a leading expert on the life and works of Claudio Monteverdi, and is the author of the books "Duet and Dialogue in the Age of Monteverdi" (Ann Arbor,
Michigan: University Microfilms International, 1982) "Monteverdi, 'Orfeo' " (London: Cambridge University Press, 1986), "Monteverdi, Vespers (1610)" (Cambridge
University Press, 1997), and "The Cambridge Companion to Monteverdi" (with Richard Wistreich, Cambridge University Press, 2007).

 For five years he was co-editor of the journal "Music & Letters".

 He currently serves on the board of the Birmingham Early Music Festival and is head of the music history department at the University of Birmingham.
</item 4>

<item 5>
Symphony in C (Bizet)
The Symphony in C is an early work by the French composer Georges Bizet.

 According to "Grove's Dictionary", the symphony "reveals an extraordinarily accomplished talent for a 17-year-old student, in melodic invention, thematic
handling and orchestration."

 Bizet started work on the symphony in C major on 29 October 1855, four days after turning 17, and finished it roughly a month later.

 It was written while he was studying at the Paris Conservatoire under the composer Charles Gounod, and was evidently a student assignment.

 Bizet showed no apparent interest in having it performed or published, and the piece was never played in his lifetime.

 He used certain material from the symphony in later works, however.

 There is no mention of the work in Bizet's letters, and it was unknown to his earlier biographers.

 His widow, Geneviève Halévy (1849–1926), gave the manuscript to Reynaldo Hahn, who left it along with other papers to the archives of the conservatory library,
where it was found in 1933 by Jean Chantavoine.
</item 5>

<item 6>
L'Arlésienne (Bizet)
Georges Bizet composed L'Arlésienne as incidental music to Alphonse Daudet's play of the same name, usually translated as "The Girl from Arles".

 It was first performed on 1 October 1872 at the Vaudeville Theatre (now a cinema known as the Gaumont Opéra).

 Bizet's music consists of 27 numbers (some only a few bars) for voice, chorus, and small orchestra, ranging from short solos to longer entr'actes. Bizet
himself played the harmonium backstage at the premiere performance.
</item 6>

<item 7>
Roma Symphony (Bizet)
The Symphony in C "Roma" is the second of Georges Bizet's symphonies.

 Unlike his first symphony, also in C major, which was written quickly at the age of 17, "Roma" was written over an eleven-year span, between the ages of 22 and
33 (he died at age 36).

 Bizet was never fully satisfied with it, subjecting it to a number of revisions, but died before finishing his definitive version.

 All four movements were performed in his lifetime, but never all on the same occasion.

 The full symphony in its latest revision was premiered in 1875, after his death.

 It is perhaps because of Bizet's dissatisfaction that the work is often said to be "unfinished".

 However, in the form in which it exists today, it is a complete work and is fully scored.

 It has been recorded a number of times but is not often heard on the concert platform.
</item 7>

<item 8>
L'Orfeo discography
These lists show the audio and visual recordings of the opera "L'Orfeo" by Claudio Monteverdi.

 The opera was first performed in Mantua in 1607, at the court of Duke Vincenzo Gonzaga, and is one of the earliest of all operas.

 The first recording of "L'Orfeo" was issued in 1939, a freely adapted version of Monteverdi's music edited by Giacomo Benvenuti, given by the orchestra of La
Scala Milan conducted by Ferrucio Calusio.

 In 1949 the Berlin Radio Orchestra under Helmut Koch recorded the complete opera, on long-playing records (LPs).

 The advent of LP recordings was, as Harold Schonberg later wrote, an important factor in the postwar revival of interest in Renaissance and Baroque music, and
from the mid-1950s recordings of "L'Orfeo" have been issued on many labels.

 Koch's landmark version was reissued in 1962, when it was compared unfavourably with others that had by then been issued.
</item 8>

<item 9>
Symphony in C (Bizet)
Soon thereafter, Bizet's first British biographer Douglas Charles Parker (1885–1970) showed the manuscript to the conductor Felix Weingartner, who led the first
performance in Basel, Switzerland, on 26 February 1935.
</item 9>

<item 10>
Giuseppe Verdi
Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
</item 10>
=== END ITEM id=d3792a7f5589499c ===

=== ITEM id=77aaf4a65b9c4769 ===
Gold sentences and facts:
Sentence 0: While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and Zoot Sims.
  Fact 0.0: The musician studied at the University of Michigan.
  Fact 0.1: While studying at the University of Michigan, the musician played with visiting musician Dizzy Gillespie.
  Fact 0.2: While studying at the University of Michigan, the musician played with visiting musician Miles Davis.
  Fact 0.3: While studying at the University of Michigan, the musician played with visiting musician Thad Jones.
  Fact 0.4: While studying at the University of Michigan, the musician played with visiting musician Billy Mitchell.
  Fact 0.5: After that, the musician played with Sonny Stitt, Count Basie and Al McKibbon.
  Fact 0.6: After that, the musician played with Cannonball Adderley, Percy Heath, Philly Joe Jones and Lou Donaldson.
  Fact 0.7: After that, the musician played with Donald Byrd, Kenny Dorham and Zoot Sims.
Sentence 1: Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.
  Fact 1.0: Miles Dewey Davis III was born on May 26, 1926.
  Fact 1.1: Miles Dewey Davis III died on September 28, 1991.
  Fact 1.2: Miles Dewey Davis III was an American jazz trumpeter.
  Fact 1.3: Miles Dewey Davis III was an American jazz bandleader.
  Fact 1.4: Miles Dewey Davis III was an American jazz composer.

Retrieved items:
<item 1>
Marty Marsala
Marty Marsala (2 April 1909 – 27 April 1975) was an American jazz trumpeter born in Chicago, perhaps best known for working from 1926-1946 with his brother Joe
Marsala in a big band in New York City and Chicago.

 He had also toured with various artists, such as Chico Marx and Miff Mole, to name a few.

 During the 1940s Marsala was a celebrated West Coast jazz trumpeter, commuting back and forth from Chicago to San Francisco frequently.

 In various club settings Marsala shared stages with Earl Hines and Sidney Bechet.
</item 1>

<item 2>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 2>

<item 3>
Don Ellis
Donald Johnson Ellis (July 25, 1934 – December 17, 1978) was an American jazz trumpeter, drummer, composer, and bandleader.

 He is best known for his extensive musical experimentation, particularly in the area of time signatures. Later in his life he worked as a film composer,
contributing a score to 1971's "The French Connection" and 1973's "The Seven-Ups".
</item 3>

<item 4>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 4>

<item 5>
Miles Davis
Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.

 He is among the most influential and acclaimed figures in the history of jazz and 20th century music.

 Davis adopted a variety of musical directions in his five-decade career which kept him at the forefront of a number of major stylistic developments in jazz.
</item 5>

<item 6>
Thad Jones
Thaddeus Joseph Jones (March 28, 1923 – August 20, 1986) was an American jazz trumpeter, composer, and bandleader who has been called "one of the all-time
greatest jazz trumpet soloists."
</item 6>

<item 7>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 7>

<item 8>
Louis Smith (gymnast)
Louis Antoine Smith, MBE (born 22 April 1989) is a British artistic gymnast.

 He received a bronze medal and a silver medal on the pommel horse at the 2008 Beijing Olympics and the 2012 London Olympics respectively, with the former
marking the first time a British gymnast had placed in an Olympic event since 1928.

 He followed this up with a second consecutive silver medal on the pommel horse at the 2016 Rio Olympics, finishing behind teammate Max Whitlock.

 Smith was part of the Great Britain team that took the bronze in the men's artistic team all-around at the 2012 London Olympics.

 He is the only British gymnast to win Olympic medals in three separate Games.

 In 2015 he became the European champion on Pommel Horse.
</item 8>

<item 9>
Paul Smith (defensive end)
Paul Edward Smith (August 13, 1945 – March 14, 2000) was a collegiate and professional American football defensive end.

 Smith was selected in the 1968 Common Draft by the American Football League's Denver Broncos and played in the National Football League for the Broncos and
Washington Redskins.
</item 9>

<item 10>
Louis Smith (musician)
While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before
going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and
Zoot Sims.
</item 10>
=== END ITEM id=77aaf4a65b9c4769 ===

=== ITEM id=7a975d90492f7a6b ===
Gold sentences and facts:
Sentence 0:  It consists of three shots, two Sourz Apple and one Cherry Brandy.
  Fact 0.0: The described drink consists of three shots.
  Fact 0.1: The described drink has two shots of Sourz Apple.
  Fact 0.2: The described drink has one shot of Cherry Brandy.
Sentence 1:  Each shot is taken from an average shot glass which is around 1.5 ounces (roughly 45 ml).
  Fact 1.0: Each shot is taken from an average shot glass.
  Fact 1.1: An average shot glass holds around 1.5 ounces (roughly 45 ml).
Sentence 2: The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.
  Fact 2.0: The flaming volcano is a large tropical group cocktail.
  Fact 2.1: The flaming volcano is made of rum, brandy, pineapple juice, orange juice and almond syrup.
  Fact 2.2: The flaming volcano is sometimes made with other ingredients.
Sentence 3:  It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about 32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.
  Fact 3.0: The flaming volcano is usually a multi-user drink.
  Fact 3.1: The flaming volcano is served to a group in a special ritual vessel known as a volcano bowl.
  Fact 3.2: A volcano bowl is a decorative ceramic bowl.
  Fact 3.3: A volcano bowl typically has a capacity of about 32 USoz.
  Fact 3.4: A volcano bowl is designed with a rising central hub feature.
  Fact 3.5: The rising central hub feature of a volcano bowl usually resembles a crude volcano cone.

Retrieved items:
<item 1>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 1>

<item 2>
Feuerzangenbowle
Feuerzangenbowle (] ) is a traditional German alcoholic drink for which a rum-soaked sugarloaf is set on fire and drips into mulled wine.

 It is often part of a Christmas or New Year's Eve tradition.

 The name translates literally to "fire-tongs punch".
</item 2>

<item 3>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 3>

<item 4>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 4>

<item 5>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 5>

<item 6>
Flaming Doctor Pepper
A Flaming Doctor Pepper is a flaming cocktail said to taste like the soft drink Dr Pepper, although Dr Pepper is not one of its ingredients.
</item 6>

<item 7>
Tweed Volcano
Tweed Volcano is a partially eroded Early Miocene shield volcano located in northeastern New South Wales, which formed when this region of Australia passed over
the East Australia hotspot around 23 million years ago.

 Mount Warning, Lamington Plateau and the Border Ranges between New South Wales and Queensland are among the remnants of this volcano that was originally over
100 km in diameter and nearly twice the height of Mount Warning today, at 1156 m .

 Despite its size, Tweed Volcano was not a supervolcano; other shield volcanoes - such as on Hawaii - are much larger.

 In the 23 million years since the volcano was active, erosion has been extensive, forming a large erosion caldera around the volcanic plug of Mount Warning.

 Its erosion caldera is the largest in the Southern Hemisphere.
</item 7>

<item 8>
Hotlum Glacier
The Hotlum Glacier is a glacier situated on the northeast flank of Mount Shasta, in the US state of California.

 It is the largest and most voluminous glacier in California, although not as thick or long as the nearby Whitney Glacier.

 The Hotlum Glacier flows from a large cirque on the northeast side of Mount Shasta's main summit below the Hotlum Headwall at roughly 13600 ft .

 It flows northeastward down the steep slope, forming three lobes which terminate near 10400 ft .
</item 8>

<item 9>
Apple-Kneel
The Apple-Kneel is a cocktail.

 It consists of three shots, two Sourz Apple and one Cherry Brandy.

 Each shot is taken from an average shot glass which is around 1.5 ounces (roughly 45 ml).

 Firstly, you put two shots of Sourz Apple into a glass and then put in one shot of Cherry Brandy in and mix.

 It is customary to sip this cocktail.

 It can be served 'neat' or 'on the rocks'.
</item 9>

<item 10>
Volcanic bomb
A volcanic bomb is a mass of molten rock (tephra) larger than 64 mm (2.5 inches) in diameter, formed when a volcano ejects viscous fragments of lava during an
eruption.

 They cool into solid fragments before they reach the ground.

 Because volcanic bombs cool after they leave the volcano, they are extrusive igneous rocks.

 Volcanic bombs can be thrown many kilometres from an erupting vent, and often acquire aerodynamic shapes during their flight.

 Bombs can be extremely large; the 1935 eruption of Mount Asama in Japan expelled bombs measuring 5–6 m in diameter up to 600 m from the vent.

 Volcanic bombs are a significant volcanic hazard, and can cause severe injuries and death to people in an eruption zone.

 One such incident occurred at Galeras volcano in Colombia in 1993; six people near the summit were killed and several seriously injured by lava bombs when the
volcano erupted unexpectedly.
</item 10>
=== END ITEM id=7a975d90492f7a6b ===

=== ITEM id=f462d0178e891b84 ===
Gold sentences and facts:
Sentence 0: Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.
  Fact 0.0: Edward Benjamin Britten, Baron Britten, lived from 22 November 1913 to 4 December 1976.
  Fact 0.1: Edward Benjamin Britten, Baron Britten, was an English composer.
  Fact 0.2: Edward Benjamin Britten, Baron Britten, was an English conductor.
  Fact 0.3: Edward Benjamin Britten, Baron Britten, was an English pianist.
Sentence 1: Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.
  Fact 1.0: Béla Viktor János Bartók lived from 25 March 1881 to 26 September 1945.
  Fact 1.1: Béla Viktor János Bartók was a Hungarian composer.
  Fact 1.2: Béla Viktor János Bartók was a Hungarian pianist.
  Fact 1.3: Béla Viktor János Bartók was a Hungarian ethnomusicologist.

Retrieved items:
<item 1>
Benjamin Britten
Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.

 He was a central figure of 20th-century British classical music, with a range of works including opera, other vocal music, orchestral and chamber pieces.

 His best-known works include the opera "Peter Grimes" (1945), the "War Requiem" (1962) and the orchestral showpiece "The Young Person's Guide to the Orchestra"
(1945).
</item 1>

<item 2>
Graph query row:
person: Benjamin Britten
nationality: English
associated_country: None
country: None
</item 2>

<item 3>
List of string quartets by Béla Bartók
1 and 28 both owe a great deal to Bartók's quartets (; ), György Ligeti, whose two string quartets both owe a great deal to Bartók’s quartets (; ), Bruno
Maderna , George Perle, who credits the Bartók Fourth and Fifth Quartets as precedents for his use of arrays of chords related to one another by different types
of symmetry , Walter Piston (; ), Kim Dzmitrïyevich Tsesakow , Wilfried Westerlinck , Stefan Wolpe, who explained in a public lecture how he had derived ideas
from Bartók’s Fourth Quartet , and Xu Yongsan .
</item 3>

<item 4>
Graph query row:
person: Béla Bartók
nationality: Hungarian
associated_country: Hungary
country: None
</item 4>

<item 5>
String Quartet No. 6 (Bartók)
The String Quartet No. 6, Sz.

 114, BB 119, was the final string quartet that Béla Bartók wrote before his death.

 It was begun in August 1939 in Saanen, Switzerland, where Bartók was a guest of his patron, the conductor Paul Sacher.

 Shortly after he completed the Divertimento for String Orchestra on the 17th, he started on a commission for his friend, the violinist Zoltán Székely.

 Székely was acting as intermediary for the "New Hungarian Quartet", who had given the Budapest premiere of the String Quartet No. 5.

 With the outbreak of World War II and his mother's illness, Bartók returned to Budapest, where the quartet was finished in November.

 After his mother's death, Bartók decided to leave with his family for the United States.
</item 5>

<item 6>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 6>

<item 7>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 7>

<item 8>
List of compositions by Béla Bartók
This aspires to be a complete list of compositions by Béla Bartók.

 The catalogue numbering by András Szőllősy (Sz.)

, László Somfai (BB.)

 and Denijs Dille (DD.)

 are provided, as well as Bartók's own opus numbers.

 Note that Bartók started three times anew with opus numbers, here indicated with "(list 1)", "(list 2)" and "(list 3)" respectively.

 The pieces from the third listing are by far best known; opus lists 1 and 2 are early works.

 The year of composition and instrumentation (including voice) are included.

 See the main article on Béla Bartók for more details.
</item 8>

<item 9>
Suite, Op. 14 (Bartók)
The Suite, Op. 14, Sz.

 62, BB 70 is a piece for solo piano written by Béla Bartók.

 It was written in February 1916, published in 1918, and debuted by the composer on April 21, 1919, in Budapest.

 The Suite is one of Bartók's most significant works for piano, only comparable with his 1926 Piano Sonata.

 Though much of Bartók's work makes frequent use of Eastern European folk music, this suite is one of the few pieces without melodies of folk origin.

 However, Romanian, Arabic, and North African rhythmic influences can still be found in some movements.

 Originally intending the suite to be a five-movement work, Bartók later decided against the idea and discarded the second movement, the "Andante", which was
published only posthumously in the October 1955 issue of "Új Zenei Szemle" (New Musical Review).
</item 9>

<item 10>
List of string quartets by Béla Bartók
The Hungarian composer Béla Bartók wrote six string quartets, each for the usual forces of two violins, viola and cello.

 Notable composers who have been influenced by them include Benjamin Britten, particularly in the Sonata in C for Cello and Piano (; ), Elliott Carter, who
refers in the opening of his own First String Quartet to Bartók’s Sixth Quartet , Chen Yi , Edison Denisov, whose Second Quartet is closely related to Bartók’s
Fifth Quartet , Franco Donatoni, who was deeply impressed when he heard a broadcast of Bartók's Fourth Quartet , Robert Fripp, who mentions them as an influence
upon King Crimson , Miloslav Ištvan , György Kurtág, whose Opp.
</item 10>
=== END ITEM id=f462d0178e891b84 ===

=== ITEM id=99f6d2f701165816 ===
Gold sentences and facts:
Sentence 0: Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.
  Fact 0.0: Disteghil Sar is also called Distaghil Sar.
  Fact 0.1: Disteghil Sar is the highest mountain in the Hispar Muztagh.
  Fact 0.2: The Hispar Muztagh is a subrange of the Karakoram mountain range.
  Fact 0.3: Disteghil Sar is in Gilgit-Baltistan.
Sentence 1:  The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
  Fact 1.0: Disteghil Sar has a top ridge about 3 km long.
  Fact 1.1: The top ridge of Disteghil Sar is above 7400 m.
  Fact 1.2: Disteghil Sar has three distinct summits.
  Fact 1.3: The (north)west summit of Disteghil Sar is 7885 m high.
  Fact 1.4: The central summit of Disteghil Sar is 7760 m high.
  Fact 1.5: The (south)east summit of Disteghil Sar is 7696 m or 7535 m high.
Sentence 2:  Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
  Fact 2.0: The summit of Disteghil Sar has an elevation of 7295 m.
  Fact 2.1: Disteghil Sar is the highest peak in the Yengisogat subrange of the Karakoram.

Retrieved items:
<item 1>
Yengisogat
The Yengisogat range (音苏盖提 ), also known as the Wesm Mountains, is a Chinese subrange of the Karakoram mountain range.

 It lies north of the Baltoro Muztagh, home of the eight-thousanders of the Karakoram.

 The highest peak is Huangguan Shan, or Crown Peak, 7,265 m (23,835 ft) (also sometimes given as 7,295 m/23,934 ft).
</item 1>

<item 2>
Kezhen Peak
Kezhen Peak, also known as Karpogo Sar, is a mountain in the Karakoram mountain range.

 It is located in the Xinjiang Autonomous Region of China.
</item 2>

<item 3>
Passu Sar
Passu Sar (Urdu: ‎ ; or Passu Sar, Passu I) is a mountain peak in the Batura Muztagh, a sub-range of the Karakoram mountain range, located in the Gilgit
District of Gilgit-Baltistan, Pakistan, west of the Hunza Valley.

 It is the high point of the Passu massif, which also includes Passu Diar (or "Passu East", "Pasu II").

 The peak lies on the main ridge of the Batura Muztagh, about 7 km (4 mi) east of Batura Sar.
</item 3>

<item 4>
The Crown (mountain)
The Crown, also known as Huang Guan Shan and sometimes Crown Peak, is a mountain in the Karakoram mountain range in China.

 It is located in the Xinjiang Uyghur Autonomous Region of China.

 Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
</item 4>

<item 5>
Yutmaru Sar
Yutmaru Sar is a mountain in the Hispar mountain range, a subrange of the Karakoram.

 At an elevation of 7283 m it is the 88th highest mountain in the world.

 Yutmaru Sar is located in Gilgit-Baltistan, Pakistan.

 It was first climbed in 1980.
</item 5>

<item 6>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 6>

<item 7>
Kanjut Sar
Kanjut Sar (Urdu: کنجت سر‎ ) or Kunjudh Sar as pronounced in "Wakhi" is a mountain located in the Hispar Muztagh, a subrange of the Karakoram mountain range.

 Kunjudh Sar in wakhi language mean that which overlooks Kunjudh, or above Kunjudh, while Khujudh is the wakhi name for Lower Hunza.

 It is the 26th highest mountain on Earth and the 11th highest in Pakistan.
</item 7>

<item 8>
Momhil Sar
Momhil Sar, or Mumhail Sar as pronounced in Wakhi is at 7343 m above sea level, is the 64th highest mountain peak in the world.

 Mumhail Sar in Wakhi means the mountain that overlooks or is above Grandmother's cattle pen or paddock.

 It is situated in the Hispar Muztagh subrange of the Karakoram range, a few kilometres to the north-west of its parent peak Trivor.
</item 8>

<item 9>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 9>

<item 10>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 10>
=== END ITEM id=99f6d2f701165816 ===

=== ITEM id=179065148f89596d ===
Gold sentences and facts:
Sentence 0: The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.
  Fact 0.0: The Shins are an American indie rock band.
  Fact 0.1: The Shins are from Albuquerque, New Mexico.
  Fact 0.2: The Shins were formed in 1996.
Sentence 1: Paramore is an American rock band from Franklin, Tennessee, formed in 2004.
  Fact 1.0: Paramore is an American rock band.
  Fact 1.1: Paramore is from Franklin, Tennessee.
  Fact 1.2: Paramore was formed in 2004.

Retrieved items:
<item 1>
Graph query row:
band: The Shins
formation_date: 1996
</item 1>

<item 2>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 2>

<item 3>
Graph query row:
band: Paramore
formation_date: 2004
</item 3>

<item 4>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 4>

<item 5>
Singles Club (EP)
Singles Club is an EP released by American rock band Paramore.

 The songs were released as promotional singles between October and December 2011, culminating in the release of a box set containing the three constituent
songs plus "Monster", which was recorded during the same sessions.

 The EP and box set were released on December 14, 2011 on Paramore's website.

 The EP is the first release of new material by Paramore not to include band members Josh and Zac Farro.
</item 5>

<item 6>
Paramore (album)
Paramore is the self-titled fourth studio album by American rock band Paramore.

 It was released on April 5, 2013, through Fueled by Ramen as a follow-up to "Brand New Eyes" (2009).

 It is their first full-length album released after the departure of co-founders Josh and Zac Farro in 2010.

 Recorded between April and November 1, 2012, the album was described by the band as being a "statement" and a reintroduction of the band to the world and to
themselves.

 It is the only album without Zac Farro on drums, and the final album recorded with bassist Jeremy Davis before he left the band in 2015.
</item 6>

<item 7>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 7>

<item 8>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 8>

<item 9>
List of songs recorded by Paramore
American rock band Paramore have recorded songs for five studio albums, a box set, an extended play and two soundtrack albums.

 In 2002, at age 13, vocalist Hayley Williams moved to Franklin, Tennessee, where she met brothers Josh Farro and Zac Farro.

 The band was officially formed by Josh Farro (lead guitar and backing vocals), Zac Farro (drums), Jeremy Davis (bass guitar) and Williams (lead vocals) in
2004, with the later addition of Williams' neighbor Jason Bynum (rhythm guitar).

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling" that year.

 Three singles were released to promote the album.
</item 9>

<item 10>
Paramore discography
The American rock band Paramore has released five studio albums, three extended plays, two live albums, eighteen singles, one video album, and nineteen music
videos.

 The band was formed in Franklin, Tennessee, in 2004 by lead vocalist Hayley Williams with guitarists Josh Farro and Taylor York, bassist Jeremy Davis, and
drummer Zac Farro.

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling".

 Three singles were released from the album, but none of them charted.

 The album did not chart in the "Billboard" 200 either, although it peaked at number thirty in the "Billboard" Top Heatseekers.

 "All We Know Is Falling" received Gold certification in the United Kingdom and in July 2014 the RIAA certified the album Gold in the United States.
</item 10>
=== END ITEM id=179065148f89596d ===

=== ITEM id=091c8ef3ca9238e9 ===
Gold sentences and facts:
Sentence 0: The City Heiress is a play by Aphra Behn produced in 1682.
  Fact 0.0: The City Heiress is a play by Aphra Behn.
  Fact 0.1: The City Heiress was produced in 1682.
Sentence 1: Aphra Behn ( ; 14 December 1640?
  Fact 1.0: Aphra Behn's birth date is listed as 14 December 1640, marked with a question mark as uncertain.
Sentence 2:  (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.
  Fact 2.0: The 14 December 1640 date for Aphra Behn is a baptismal date.
  Fact 2.1: Aphra Behn died on 16 April 1689.
  Fact 2.2: Aphra Behn was a British playwright.
  Fact 2.3: Aphra Behn was a poet.
  Fact 2.4: Aphra Behn was a translator.
  Fact 2.5: Aphra Behn was a fiction writer.
  Fact 2.6: Aphra Behn was from the Restoration era.

Retrieved items:
<item 1>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 1>

<item 2>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 2>

<item 3>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 3>

<item 4>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 4>

<item 5>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 5>

<item 6>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 6>

<item 7>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 7>

<item 8>
The Heiress (1786 play)
The Heiress is a comedy play by the British playwright and soldier John Burgoyne.

 The play debuted at the Drury Lane Theatre on 14 January 1786.

 It concerns the engagement of Lord Gayville to Miss Alscrip, a fashionable woman he believes to be an heiress.

 Gayville later discovers that the woman who really stands to inherit the fortune is his true love Miss Clifford.

 The play was an enormous success, running for 31 performances in its initial season and being revived again the following year.

 The play was initially anonymous, but Burgoyne was soon widely reported to be the author and he acknowledged this after the play's debut.
</item 8>

<item 9>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 9>

<item 10>
The City Heiress
The City Heiress is a play by Aphra Behn produced in 1682.

 The play conforms to the general rules of Restoration comedy, but it also keeps Behn's own highly Royalist political point of view.
</item 10>
=== END ITEM id=091c8ef3ca9238e9 ===

=== ITEM id=d5e0cd8ebb26c6bb ===
Gold sentences and facts:
Sentence 0: Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.
  Fact 0.0: Sleepless in Seattle is a 1993 film.
  Fact 0.1: Sleepless in Seattle is an American film.
  Fact 0.2: Sleepless in Seattle is a romantic comedy-drama film.
  Fact 0.3: Sleepless in Seattle was directed by Nora Ephron.
  Fact 0.4: Sleepless in Seattle was co-written by Nora Ephron.
  Fact 0.5: Sleepless in Seattle is based on a story by Jeff Arch.
Sentence 1:  It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber, and Rita Wilson.
  Fact 1.0: Sleepless in Seattle stars Tom Hanks.
  Fact 1.1: Sleepless in Seattle stars Meg Ryan.
  Fact 1.2: Sleepless in Seattle has a supporting cast featuring Bill Pullman.
  Fact 1.3: Sleepless in Seattle has a supporting cast featuring Ross Malinger.
  Fact 1.4: Sleepless in Seattle has a supporting cast featuring Rob Reiner.
  Fact 1.5: Sleepless in Seattle has a supporting cast featuring Rosie O'Donnell.
  Fact 1.6: Sleepless in Seattle has a supporting cast featuring Gaby Hoffmann.
  Fact 1.7: Sleepless in Seattle has a supporting cast featuring Victor Garber.
  Fact 1.8: Sleepless in Seattle has a supporting cast featuring Rita Wilson.
Sentence 2: Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.
  Fact 2.0: Roseann O'Donnell was born on March 21, 1962.
  Fact 2.1: Roseann O'Donnell is an American comedian.
  Fact 2.2: Roseann O'Donnell is an American actress.
  Fact 2.3: Roseann O'Donnell is an American author.
  Fact 2.4: Roseann O'Donnell is an American television personality.

Retrieved items:
<item 1>
Kate O'Beirne
Kate Walsh O'Beirne (September 23, 1949 – April 23, 2017) was the President of National Review Institute.

 She was the Washington editor of "National Review".

 Her column, "Bread and Circuses," covered Congress, politics, and U.S. domestic policy.
</item 1>

<item 2>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 2>

<item 3>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 3>

<item 4>
Barry Sandler
Barry Sandler (born February 23, 1947 in Buffalo, New York) is an American screenwriter and film producer.

 His career has spanned several decades, with the 1980s being his most prolific.

 The openly gay Sandler is perhaps best known for writing the 1982 film "Making Love", the first mainstream Hollywood film to deal seriously with issues of
homosexuality and coming out.

 Sandler discussed "Making Love" in the 1995 documentary film "The Celluloid Closet".
</item 4>

<item 5>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 5>

<item 6>
Matthew Perry
Matthew Langford Perry (born August 19, 1969) is a Canadian-American actor, best known for his role as Chandler Bing on the long-running NBC television sitcom
"Friends", as well as his portrayal of Ron Clark in the 2006 television movie "The Ron Clark Story".
</item 6>

<item 7>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 7>

<item 8>
Rosie O'Donnell
Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.

 She has been a magazine editor and continues to be a celebrity blogger, a lesbian rights activist, a television producer, and a collaborative partner in the
LGBT family vacation company, R Family Vacations.
</item 8>

<item 9>
Carrie Fisher
Carrie Frances Fisher (October 21, 1956 – December 27, 2016) was an American actress, writer, and humorist.

 Fisher was known for playing Princess Leia in the "Star Wars" films; a role for which she was nominated for three Saturn Awards.

 Her other film roles included "Shampoo" (1975), "The Blues Brothers" (1980), "Hannah and Her Sisters" (1986), "The 'Burbs" (1989), "When Harry Met Sally..."
(1989), "Soapdish" (1991) and "The Women" (2008 film).

 She was nominated twice for the Primetime Emmy Award for Outstanding Guest Actress in a Comedy Series for her performances on the television series "30 Rock"
and "Catastrophe".

 She was posthumously made a Disney Legend in 2017.
</item 9>

<item 10>
Reese Witherspoon
Laura Jeanne Reese Witherspoon ( ; born March 22, 1976) is an American actress, producer, and entrepreneur.

 Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for
which she was nominated for a Young Artist Award.

 Following breakout roles in "" (1992) and "Jack the Bear" (1993), she starred in the comedy-drama "Pleasantville" (1998), for which she won the Young Hollywood
Award for Breakthrough Performance.

 Her leading role of Tracy Flick in "Election" (1999) was nominated for a Golden Globe Award.
</item 10>
=== END ITEM id=d5e0cd8ebb26c6bb ===

=== ITEM id=7b441106f7339f47 ===
Gold sentences and facts:
Sentence 0: Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
  Fact 0.0: Alan Crosland was born on August 10, 1894.
  Fact 0.1: Alan Crosland died on July 16, 1936.
  Fact 0.2: Alan Crosland was an American stage actor.
  Fact 0.3: Alan Crosland was an American film director.
Sentence 1: Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.
  Fact 1.0: Kurt Voss was born Kurt Christopher Peter Wössner.
  Fact 1.1: Kurt Voss is an American film director.
  Fact 1.2: Kurt Voss is an American screenwriter.
  Fact 1.3: Kurt Voss is an American musician-songwriter.

Retrieved items:
<item 1>
Midnight Alibi
Midnight Alibi is a 1934 sound film directed by Alan Crosland, produced by First National Pictures, distributed by [[Warner Bros].]

 and starring [[Richard Barthelmess]].

 Midnight Alibi is an adaptation of [[Damon Runyon]]'s 1933 short story "The Old Doll's House".
</item 1>

<item 2>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 2>

<item 3>
Araby Lockhart
Araby Lockhart (born December 4, 1926) is a Canadian stage and television actress, best known for her performances in the films "Capote" and "Police Academy"
and her stage performances as a member of Hart House Theatre and the Straw Hat Players.

 Lockhart has also served as President of the Actors' Fund of Canada.
</item 3>

<item 4>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 4>

<item 5>
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
</item 5>

<item 6>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 6>

<item 7>
General Crack
General Crack is a 1930 American Pre-Code part-talkie historical costume melodrama with Technicolor sequences which was directed by Alan Crosland and produced
and distributed by Warner Bros. It was filmed and premiered in 1929, and released early in 1930.

 It stars John Barrymore in his first full-length all-talking feature.

 The film would prove to be Crosland and Barrymore's last historical epic together.
</item 7>

<item 8>
Lady Tubbs
Lady Tubbs is a 1935 American comedy film directed by Alan Crosland and written by Barry Trivers.

 The film stars Alice Brady, Douglass Montgomery, Anita Louise, Alan Mowbray, June Clayworth and Hedda Hopper.

 The film was released on July 2, 1935, by Universal Pictures.
</item 8>

<item 9>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 9>

<item 10>
Contraband (1925 film)
Contraband is a lost 1925 silent film produced by Famous Players-Lasky and distributed by Paramount Pictures.

 Alan Crosland directed and Lois Wilson stars.

 The film is taken from a novel, "Contraband", by Charles Buddington Kelland.

 The last film directed by Alan Crosland the cooperation with distributor Paramount Pictures.
</item 10>
=== END ITEM id=7b441106f7339f47 ===

=== ITEM id=8aec0d6c9271434b ===
Gold sentences and facts:
Sentence 0: The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.
  Fact 0.0: The Dundee Canal was an industrial canal.
  Fact 0.1: The Dundee Canal was in Clifton and Passaic.
  Fact 0.2: Clifton and Passaic are in Passaic County, New Jersey.
Sentence 1:  It was built between 1858 and 1861 and ran parallel to the Passaic River.
  Fact 1.0: The Dundee Canal was built between 1858 and 1861.
  Fact 1.1: The Dundee Canal ran parallel to the Passaic River.
Sentence 2:  It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.
  Fact 2.0: It runs parallel to the coast.
  Fact 2.1: It is separated from the Atlantic Ocean by barrier beaches.
  Fact 2.2: The barrier beaches include Palm Beach Island.
Sentence 3:  The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
  Fact 3.0: The lagoon is connected to the Atlantic Ocean.
  Fact 3.1: The lagoon is connected to the Atlantic Ocean by two inlets.
  Fact 3.2: The two inlets are permanent.
  Fact 3.3: The two inlets are man-made.

Retrieved items:
<item 1>
Lake Osborne
Lake Osborne, Florida, USA is a 378-acre (152.9 hectares) lake that is part of a system of once natural freshwater lakes lying along the western slope of the
coastal ridge in Palm Beach County just west of the Florida Intracoastal Waterway and Atlantic Ocean.

 It is located within the C-16 drainage basin which occupies approximately 40,031 acres of land (16,200 hectares).

 Five drainage canals discharge directly to Lake Osborne, and Lake Osborne discharges via the E-4 canal to the C16 and C51 canals to the Lake Worth Lagoon.

 The lake is bordered on the west by John Prince Memorial Park, and on the east be the City of Lake Worth.
</item 1>

<item 2>
South Lake Worth Inlet
The South Lake Worth Inlet, also known as the Boynton Inlet, is an artificial cut through a barrier beach connecting the south end of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 The inlet is 130 ft wide and 6 ft and 12 ft deep.
</item 2>

<item 3>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 3>

<item 4>
Lake Worth Lagoon
The Lake Worth Lagoon is a lagoon located in Palm Beach County, Florida.

 It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.

 The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
</item 4>

<item 5>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 5>

<item 6>
Lake Worth, Florida
Lake Worth is a city in Palm Beach County, Florida, United States, which takes its name from the body of water along its eastern border, originally called "Lake
Worth", and now generally known as the Lake Worth Lagoon.

 The lake itself was named for General William J. Worth, who led U.S. forces during the last part of the Second Seminole War.

 s of 2010 , the population estimated by the U.S. Census Bureau was 34,910.

 The city is included in the Miami-Fort Lauderdale-Pompano Beach Metropolitan Area, which is home to approximately 5,563,857 people.
</item 6>

<item 7>
The entries cover historic U.S. canals, waterways, water systems, and sites. Transport canals include Beaver–Erie (136 miles), Delaware (60), Whitewater (76),
and Susquehanna–Tidewater (43), plus interstate and regional routes; many declined with railroad competition. The Dismal Swamp Canal, opened in 1805 and closed
in 2016, served the Intracoastal Waterway; nearby park and refuge protect the swamp.
</item 7>

<item 8>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 8>

<item 9>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 9>

<item 10>
Lake Worth Inlet
The Palm Beach Inlet, also known as the Lake Worth Inlet is an artificial cut through a barrier island connecting the northern part of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 It is bordered by the town of Palm Beach on the south, and by the town of Palm Beach Shores to the north.

 The inlet is also the entrance channel for the Port of Palm Beach.

 Its coordinates are .
</item 10>
=== END ITEM id=8aec0d6c9271434b ===

=== ITEM id=c62238fa5477219f ===
Gold sentences and facts:
Sentence 0: Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous family.
  Fact 0.0: Homer Jay Simpson is a fictional character.
  Fact 0.1: Homer Jay Simpson is the main protagonist of the American animated television series 'The Simpsons'.
  Fact 0.2: Homer Jay Simpson is the patriarch of the Simpson family, the family the show is named after.
Sentence 1:  He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April 19, 1987.
  Fact 1.0: Homer Simpson is voiced by Dan Castellaneta.
  Fact 1.1: Homer Simpson first appeared on television along with the rest of his family.
  Fact 1.2: Homer Simpson's first television appearance was in the short 'Good Night' from 'The Tracey Ullman Show'.
  Fact 1.3: Homer Simpson's first television appearance was on April 19, 1987.
Sentence 2: Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.
  Fact 2.0: Daniel Louis Castellaneta was born October 29, 1957.
  Fact 2.1: Daniel Louis Castellaneta is an American actor.
  Fact 2.2: Daniel Louis Castellaneta is a voice actor.
  Fact 2.3: Daniel Louis Castellaneta is a comedian.
  Fact 2.4: Daniel Louis Castellaneta is a screenwriter.

Retrieved items:
<item 1>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 1>

<item 2>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 2>

<item 3>
Ned Flanders
Nedward "Ned" Flanders, Jr. is a recurring fictional character in the animated television series "The Simpsons".

 He is voiced by Harry Shearer, and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 He is the good-natured, cheery next-door neighbor to the Simpson family and is generally loathed by Homer Simpson.

 A devout Evangelical Christian with an annoyingly perfect family, he is among the friendliest and most compassionate of Springfield's residents and is
generally considered a pillar of the Springfield community.
</item 3>

<item 4>
Grampa Simpson
Abraham Jedediah Simpson II, often known as Grampa, is a fictional character in the animated television series "The Simpsons".

 He made his first appearance in the episode entitled "Grampa and the Kids", a Simpsons short on "The Tracey Ullman Show".

 Voiced by Dan Castellaneta, he is the father of Homer Simpson and the grandfather of Bart, Lisa and Maggie Simpson.

 In the 1000th issue of "Entertainment Weekly", Grampa was selected as the Grandpa for "The Perfect TV Family".
</item 4>

<item 5>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 5>

<item 6>
Citizen Joe
"Citizen Joe" is the fifteenth episode for season eight of the Canadian-American military science fiction television series "Stargate SG-1".

 The episode features known voice actor Dan Castellaneta, who voices Homer Simpson in "The Simpsons".

 The episode was written by executive producer Robert C. Cooper, the episode was directed by Andy Mikita.

 The episode received a below average Nielsen household rating and received no syndication rating to compare.

 The episode got strong reviews from major media publishers worldwide.
</item 6>

<item 7>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 7>

<item 8>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 8>

<item 9>
Homer to the Max
"Homer to the Max" is the thirteenth episode of "The Simpsons"<nowiki>'</nowiki> tenth season.

 It originally aired on the Fox network in the United States on February 7, 1999.

 In the episode, Homer discovers that a new television show, "Police Cops", has a hero also named Homer Simpson.

 He is delighted with the positive attention he receives because of his name, but when the television character is rewritten from a hero to a bumbling idiot, he
is mocked and taunted, so he changes his name to "Max Power" to rid himself of the negative attention.

 Max gains new friends, and is forced into a protest to prevent a forest from being knocked down.

 In the end, he changes his name back to Homer Simpson.
</item 9>

<item 10>
Dan Castellaneta
Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.

 Noted for his long-running role as Homer Simpson on the animated television series "The Simpsons", he also voices many other characters for the show, including
Abraham "Grampa" Simpson, Barney Gumble, Krusty the Clown, Sideshow Mel, Groundskeeper Willie, Mayor Quimby and Hans Moleman.
</item 10>
=== END ITEM id=c62238fa5477219f ===
