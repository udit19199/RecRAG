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

=== ITEM id=92ab6963f79ff263 ===
Gold sentences and facts:
Sentence 0:  She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.
  Fact 0.0: The woman described is married to Scott Stapp.
  Fact 0.1: Scott Stapp is a former frontman of the band Creed.
  Fact 0.2: Scott Stapp is the current frontman for the band Art of Anarchy.
Sentence 1:  The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between members.
  Fact 1.0: The band's third album, “Weathered”, was released in 2001.
  Fact 1.1: Tremonti handled bass when “Weathered” was released.
  Fact 1.2: The band disbanded in 2004.
  Fact 1.3: The band disbanded in 2004 due to increasing tension between members.

Retrieved items:
<item 1>
Jaclyn Stapp
Jaclyn Nesheiwat Stapp (born July 29, 1980) is a beauty queen, philanthropist and fashion model with pageant roots in Florida and New York.

 She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.

 Her most notable titles include Mrs. Florida America 2008 and Miss New York USA 2004.

 She is executive director of The Scott Stapp With Arms Wide Open Foundation.
</item 1>

<item 2>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 2>

<item 3>
Creedence Clearwater Revival (album)
Creedence Clearwater Revival is the debut studio album by the American rock band Creedence Clearwater Revival, released in 1968.
</item 3>

<item 4>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
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
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 6>

<item 7>
Creed (band)
Creed is an American rock band formed in 1993 in Tallahassee, Florida.

 The band's best-known line-up consists of lead vocalist Scott Stapp, guitarist and vocalist Mark Tremonti, bassist Brian Marshall, and drummer Scott Phillips.

 Creed released two studio albums, "My Own Prison" in 1997 and "Human Clay" in 1999, before Marshall left the band in 2000.

 The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between
members.

 Tremonti, Marshall, and Phillips went on to found Alter Bridge while Stapp followed a solo career.
</item 7>

<item 8>
Creedence Clearwater Revisited
Creedence Clearwater Revisited is an American rock band formed in 1995 by two former members of Creedence Clearwater Revival – a much more well-known band with
a very similar name.

 It is known for playing the music that was originally made famous by Creedence Clearwater Revival.

 The two common band members are Stu Cook (bass) and Doug "Cosmo" Clifford (drums).
</item 8>

<item 9>
Creedence Clearwater Revival: Box Set
Creedence Clearwater Revival: Box Set is a box set by Creedence Clearwater Revival, released in 2001.

 It contains all of their complete studio albums, two complete live albums, and material recorded by the band under their previous names "The Golliwogs" and
"The Blue Velvets".

 In November 2013 the box set was reissued with different artwork.

 This box set spans their career, and contains all of their complete materials.
</item 9>

<item 10>
Scott Weiland and the Wildabouts
Scott Weiland and the Wildabouts (sometimes billed simply as The Wildabouts) were a rock and roll band formed sometime around 2012 by former Stone Temple Pilots
and Velvet Revolver lead singer Scott Weiland, along with bassist Tommy Baker, guitarist Jeremy Brown, and multi-instrumentalist Doug Grean.

 The Wildabouts were marred with several roster changes (including the death of guitarist Jeremy Brown) and ultimately disbanded after Weiland's sudden death on
a tour bus in late 2015.
</item 10>
=== END ITEM id=92ab6963f79ff263 ===

=== ITEM id=0211c1670614f802 ===
Gold sentences and facts:
Sentence 0: The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista Distribution.
  Fact 0.0: The Rescuers is a 1977 film.
  Fact 0.1: The Rescuers is an American animated film.
  Fact 0.2: The Rescuers is a comedy-drama adventure film.
  Fact 0.3: The Rescuers was produced by Walt Disney Productions.
  Fact 0.4: The Rescuers was first released on June 22, 1977.
  Fact 0.5: The Rescuers was first released by Buena Vista Distribution.
Sentence 1: The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.
  Fact 1.0: The Muppets is a 2011 film.
  Fact 1.1: The Muppets is an American film.
  Fact 1.2: The Muppets is a musical comedy film.
  Fact 1.3: The Muppets is the seventh theatrical film featuring the Muppets.

Retrieved items:
<item 1>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 1>

<item 2>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 2>

<item 3>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 3>

<item 4>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 4>

<item 5>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 5>

<item 6>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
</item 6>

<item 7>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 7>

<item 8>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 8>

<item 9>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 9>

<item 10>
The Muppets at Walt Disney World
The Muppets at Walt Disney World is a television special starring Jim Henson's Muppets at Walt Disney World in Lake Buena Vista, Florida.

 The special aired on NBC as part of "The Magical World of Disney" on May 6, 1990, and turned out to be the last Muppets special that Henson would work on (he
would die 10 days following its airing, on May 16).
</item 10>
=== END ITEM id=0211c1670614f802 ===

=== ITEM id=a3b2ba5cac3b0f18 ===
Gold sentences and facts:
Sentence 0:  The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
  Fact 0.0: The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part.
  Fact 0.1: Sammy Hagar played the rhythm guitar part on a Gibson acoustic.
Sentence 1: OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist Sammy Hagar.
  Fact 1.0: OU812 is pronounced 'Oh You Ate One Too'.
  Fact 1.1: OU812 is the eighth studio album by American hard rock band Van Halen.
  Fact 1.2: OU812 was released in 1988.
  Fact 1.3: OU812 is the second Van Halen album to feature vocalist Sammy Hagar.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
OU812 Tour
The OU812 Tour was a concert tour by hard rock band Van Halen.

 It was the second tour to include dates in Japan, and was the second with Sammy Hagar as vocalist (who had settled into the role more by this point).

 It followed the Monsters of Rock Tour 1988, which had formed the first part of the promotion for the "OU812" album.
</item 2>

<item 3>
Keisuke Kuwata (album)
Keisuke Kuwata is the eponymous studio album recorded by Keisuke Kuwata, a frontman of the Japanese rock band Southern All Stars.

 It was released by Taishita label under the Victor Entertainment in July 1988, shortly after the 10th anniversary of the band.

 Aside from the 1982 live recording album "Kamon Yuzo and Victor Wheels Live" released under the pseudonym Yūzō Kamon, it was first Kuwata's solo effort.
</item 3>

<item 4>
Steve Augeri is an American rock singer known for Tall Stories, Tyketto, and Journey. English multi-instrumentalist Steve Winwood performed with the Spencer
Davis Group, Traffic, and Blind Faith; his debut solo album reached UK No. 12, while two compilations document his work. His 1986 hit “Back in the High Life
Again” topped the US Adult Contemporary chart for three weeks. Pedal-steel guitarist Speedy West played with Jimmy Bryant and on recordings by Tennessee
</item 4>

<item 5>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 5>

<item 6>
Finish What Ya Started
"Finish What Ya Started" is a song by Van Halen taken from their 1988 album "OU812".

 Despite the album being seemingly complete, Eddie Van Halen came up with the riff at 2 in the morning and went down to his then-neighbor Sammy Hagar to show
it.

 Hagar let Eddie in, and the two played guitars in his balcony until they had a completed song.

 Once Eddie left, Hagar decided to write the lyrics despite being late at night.

 The theme wound up being unfulfilled sex, summed up by Hagar as "blue balls.

 In the song, Eddie Van Halen recorded his guitar part on a Fender Stratocaster plugged direct into the studio mixing console.

 The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
</item 6>

<item 7>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 7>

<item 8>
OU812
OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist
Sammy Hagar.

 Van Halen started work on the album in September 1987 and completed it in April 1988, just one month before its release.
</item 8>

<item 9>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 9>

<item 10>
The entries cover songs and recordings spanning R&B, rock, pop, and folk. Highlights include Johnny Otis’s 1958 “Willie and the Hand Jive” (US #9), later
covered by Eric Clapton (#26) and George Thorogood; Van Halen’s “Finish What Ya Started,” co-written overnight by Eddie Van Halen and Sammy Hagar; and Johnny
Otis’s “Every Beat of My Heart,” later a major hit for the Pips.
</item 10>
=== END ITEM id=a3b2ba5cac3b0f18 ===

=== ITEM id=6e756fc2ebb333c6 ===
Gold sentences and facts:
Sentence 0: Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.
  Fact 0.0: Nelson Algren was born March 28, 1909.
  Fact 0.1: Nelson Algren died May 9, 1981.
  Fact 0.2: Nelson Algren was an American writer.
Sentence 1: Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
  Fact 1.0: Nathanael West was born Nathan Weinstein.
  Fact 1.1: Nathanael West was born October 17, 1903.
  Fact 1.2: Nathanael West died December 22, 1940.
  Fact 1.3: Nathanael West was an American author.
  Fact 1.4: Nathanael West was an American screenwriter.
  Fact 1.5: Nathanael West was an American satirist.

Retrieved items:
<item 1>
A Walk on the Wild Side
A Walk on the Wild Side is a 1956 novel by Nelson Algren, most often quoted as the source for Algren's "three rules of life": "Never play cards with a man
called Doc.

 Never eat at a place called Mom's.

 Never sleep with a woman whose troubles are worse than your own."
</item 1>

<item 2>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 2>

<item 3>
Nathanael West
Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
</item 3>

<item 4>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 4>

<item 5>
The Man with the Golden Arm (novel)
The Man with the Golden Arm is a novel by Nelson Algren, published by Doubleday in November 1949.

 One of the seminal novels of post-World War II American letters, "The Man with the Golden Arm" is widely considered Algren's greatest and most enduring work.

 It won the National Book Award in 1950.
</item 5>

<item 6>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 6>

<item 7>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 7>

<item 8>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 8>

<item 9>
Haniel Long
Haniel Clark Long (March 9, 1888 – October 17, 1956) was an American poet, novelist, publisher and academic.

 He is best known for his novella, "Interlinear to Cabeza de Vaca" (1936), a fictionalized account of the true story of a Spanish conquistador in 16th century
North America.
</item 9>

<item 10>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 10>
=== END ITEM id=6e756fc2ebb333c6 ===

=== ITEM id=50bbac8b5521a5d3 ===
Gold sentences and facts:
Sentence 0: John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.
  Fact 0.0: John Kingsley Orton, known as Joe Orton, was born on 1 January 1933.
  Fact 0.1: Joe Orton died on 9 August 1967.
  Fact 0.2: Joe Orton was an English playwright.
  Fact 0.3: Joe Orton was an English author.
Sentence 1:  His public career was short but prolific, lasting from 1964 until his death three years later.
  Fact 1.0: Joe Orton's public career was short but prolific.
  Fact 1.1: Joe Orton's public career lasted from 1964 until his death.
  Fact 1.2: Joe Orton died three years after his public career began in 1964.
Sentence 2:  During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.
  Fact 2.0: During his brief public career, Joe Orton shocked audiences with his scandalous black comedies.
  Fact 2.1: During his brief public career, Joe Orton outraged audiences with his scandalous black comedies.
  Fact 2.2: During his brief public career, Joe Orton amused audiences with his scandalous black comedies.
Sentence 3:  The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
  Fact 3.0: The adjective 'Ortonesque' is sometimes used to refer to work characterised by a dark yet farcical cynicism.
  Fact 3.1: The cynicism is similarly dark yet farcical.
Sentence 4: Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary modernism, and is considered by many to be among the greatest novelists in the English language.
  Fact 4.0: Henry James, OM, was an American author.
  Fact 4.1: Henry James was born in 1843.
  Fact 4.2: Henry James died in 1916.
  Fact 4.3: Henry James was regarded as a key transitional figure between literary realism and literary modernism.
  Fact 4.4: Henry James is considered by many to be among the greatest novelists in the English language.
Sentence 5:  He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
  Fact 5.0: Henry James was the son of Henry James, Sr.
  Fact 5.1: Henry James was the brother of the renowned philosopher and psychologist William James.
  Fact 5.2: Henry James was the brother of diarist Alice James.

Retrieved items:
<item 1>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 1>

<item 2>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 2>

<item 3>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 3>

<item 4>
Henry James
Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary
modernism, and is considered by many to be among the greatest novelists in the English language.

 He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
</item 4>

<item 5>
What the Butler Saw (play)
What the Butler Saw is a farce written by the English playwright Joe Orton.

 It was premièred at the Queen's Theatre in London on 5 March 1969.

 It was Orton's final play and the second to be performed after his death, following "Funeral Games" in 1968.
</item 5>

<item 6>
Prick Up Your Ears
Prick Up Your Ears is a 1987 British film, directed by Stephen Frears, about the playwright Joe Orton and his lover Kenneth Halliwell.

 The screenplay was written by Alan Bennett, based on the biography by John Lahr.

 The film stars Gary Oldman as Orton, Alfred Molina as Halliwell, Wallace Shawn as Lahr and Vanessa Redgrave as Peggy Ramsay.
</item 6>

<item 7>
Tatsuhiro Oshiro
Tatsuhiro Ōshiro (大城 立裕 , Ōshiro Tatsuhiro , born 19 September 1925) is a Japanese novelist and playwright.

 He was awarded the Akutagawa Prize in 1967 for his novella of the same year, "The Cocktail Party", which has been adapted for the stage and made into a film.

 Ōshiro has also been an innovator of the traditional Ryukyuan narrative dance form known as kumi odori.

 Having added twenty new pieces to the repertoire, Ōshiro is credited as having "single-handedly revived the genre that originated in the 18th century" by
incorporating Okinawa shibai (dramas in the Okinawan language) and distinctive rhythms to construct a fluid, hybrid cultural identity.

 His writings have been noted for making Okinawan culture and history accessible to Japanese readership, while his more popular works have been critically
praised for "offering an acute perspective on the psychological and moral implications of war and military occupation."
</item 7>

<item 8>
The Ruffian on the Stair
The Ruffian On the Stair is a play by British playwright Joe Orton which was first broadcast on BBC Radio in August 1964.

 It is an unsympathetic yet comedic one-act portrayal of working class England, as played out by a couple and a mysterious young man who toys with their lives.

 It was based on "The Boy Hairdresser", a novel by Orton and his lover Kenneth Halliwell.

 The title and play are based on a few lines from poet and dramatist William Ernest Henley: "Madam Life's a piece in bloom, / Death goes dogging everywhere: /
She's the tenant of the room, / He's the ruffian on the stair."

 "Ruffian" is not as renowned as other works such as "Loot" and "What the Butler Saw", but it is still staged on occasion.
</item 8>

<item 9>
Prick Up Your Ears (play)
Prick Up Your Ears is a play by Simon Bent, based on the life of playwright Joe Orton.

 Produced by Sonia Friedman it opened at the Comedy Theatre in London's West End on 30 September 2009 following previews from 17 September.

 It starred Chris New as Joe Orton and Matt Lucas as Orton's lover and murderer, Kenneth Halliwell.
</item 9>

<item 10>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 10>
=== END ITEM id=50bbac8b5521a5d3 ===

=== ITEM id=8de7280857c4d12d ===
Gold sentences and facts:
Sentence 0: Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister, and for two terms President of the General Assembly of the League of Nations (1930–32).
  Fact 0.0: Nicolae Titulescu was born on March 4, 1882.
  Fact 0.1: Nicolae Titulescu died on March 17, 1941.
  Fact 0.2: Nicolae Titulescu was a well-known Romanian diplomat.
  Fact 0.3: Nicolae Titulescu was at various times a government minister.
  Fact 0.4: Nicolae Titulescu was at various times finance minister.
  Fact 0.5: Nicolae Titulescu was at various times foreign minister.
  Fact 0.6: Nicolae Titulescu served for two terms as President of the General Assembly of the League of Nations.
  Fact 0.7: Nicolae Titulescu's two terms as President of the General Assembly of the League of Nations are dated 1930–32.
Sentence 1: The League of Nations (abbreviated as LN in English, "La Société des Nations " ] abbreviated as SDN or SdN in French) was an intergovernmental organisation founded on 10 January 1920 as a result of the Paris Peace Conference that ended the First World War.
  Fact 1.0: The League of Nations is abbreviated as LN in English.
  Fact 1.1: The League of Nations is abbreviated as SDN or SdN in French.
  Fact 1.2: The French name of the League of Nations is 'La Société des Nations'.
  Fact 1.3: The League of Nations was an intergovernmental organisation.
  Fact 1.4: The League of Nations was founded on 10 January 1920.
  Fact 1.5: The League of Nations was founded as a result of the Paris Peace Conference.
  Fact 1.6: The Paris Peace Conference ended the First World War.

Retrieved items:
<item 1>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 1>

<item 2>
Remus Nicolai
Remus Nicolai (b. June 10, 1977 in Bistriţa, Romania) is a retired Romanian aerobic gymnast.

 He had a successful career winning six world championships medals (one gold, two silver and three bronze) and five European championships medals (two gold, two
silver and one bronze).

 After retiring from aerobic gymnastics he opened together with his wife (Daniela Mărănducă) a private gymnastics club in Constanţa.

 As of 2013 he trains, together with his wife, the junior artistic women's National Team of Romania at the National Olympic Center in Onesti.
</item 2>

<item 3>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 3>

<item 4>
Nicolae Titulescu
Nicolae Titulescu (] ; March 4, 1882 – March 17, 1941) was a well-known Romanian diplomat, at various times government minister, finance and foreign minister,
and for two terms President of the General Assembly of the League of Nations (1930–32).
</item 4>

<item 5>
Ioan Silviu Suciu
Ioan Silviu Suciu (born 24 November 1977 in Sibiu) is a retired Romanian artistic gymnast.

 His best event was the pommel horse.

 One of his closest rivals was his team mate and pommel horse Olympic champion Marius Urzică, who was defeated by Suciu in Ljubljana, 2004, when the latter won
the title while Urzică took 8th place.

 Suciu is an Olympic bronze medalist with the team, a silver world medalist on pommel horse and a six-time European medalist (pommel horse, vault and team).

 Suciu was one of the key team members of the Romanian gymnastics team for several years contributing to the 2004 Olympic team bronze medal and three
continental team medals (two gold and one silver).

 He also placed fourth all around at the 2004 Olympic Games.
</item 5>

<item 6>
Marius Urzică
Marius Daniel Urzică (born September 30, 1975, in Topliţa, Romania) is a Romanian gymnast.

 Urzică is an Olympic champion, a three-time world champion and a three-time European champion on pommel horse.

 He competed at three Olympic games, medaling each time on pommel horse (gold Sydney 2000, silver Atlanta 1996 and silver Athena 2004) and contributed to the
team bronze in Athens 2004.

 His unique technique and style of performance have won him the recognition as one of the greatest masters on this piece of apparatus ever, together with
Miroslav Cerar and Zoltán Magyar.

 Known as "The King of the Pommels" in 2001 he achieved the maximum score of 10.00 on this piece of apparatus at the Glasgow Grand Prix.

 Two elements in artistic gymnastics, one on pommel horse and one on parallel bars are named after him.
</item 6>

<item 7>
Ghimpețeni
Ghimpețeni is a commune in Olt County, Romania.

 It is composed of two villages, Ghimpețeni and Ghimpețenii Noi.

 These were part of Nicolae Titulescu Commune until 2004, when they were split off.
</item 7>

<item 8>
Graph query row:
organization: League of Nations
founding_date: 10 January 1920
</item 8>

<item 9>
Gheorghe Tătărescu
During his first time in office, he moved closer to King Carol II, leading an ambivalent policy toward the fascist Iron Guard and ultimately becoming
instrumental in establishing the authoritarian and corporatist regime around the National Renaissance Front.

 In 1940, he accepted the cession of Bessarabia and Northern Bukovina to the Soviet Union, and consequently had to resign.
</item 9>

<item 10>
Gheorghe Tătărescu
Gheorghe I. Tătărescu (also known as "Guță Tătărescu", with a slightly antiquated pet form of his given name; 2 November 1886 – 28 March 1957) was a Romanian
politician who served 36th Prime Minister of Romania (1934–1937; 1939–1940), three times as Minister of Foreign Affairs ("interim" in 1934 and 1938; appointed
to the office in 1945-1947), and once as Minister of War (1934).

 Representing the "young liberals" faction inside the National Liberal Party (PNL), Tătărescu began his political career as a collaborator of Ion G. Duca,
becoming noted for his anti-Communism and, in time, for his conflicts with the PNL's leader Dinu Brătianu and the Foreign Minister Nicolae Titulescu.
</item 10>
=== END ITEM id=8de7280857c4d12d ===

=== ITEM id=6eb165e6c7e7b20e ===
Gold sentences and facts:
Sentence 0: Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss is a book written in 2003 by Dr. Joel Fuhrman, which he revised in 2011.
  Fact 0.0: Eat to Live is a book.
  Fact 0.1: The full title of Eat to Live is 'Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss'.
  Fact 0.2: Eat to Live was written in 2003.
  Fact 0.3: Eat to Live was written by Dr. Joel Fuhrman.
  Fact 0.4: Dr. Joel Fuhrman revised Eat to Live in 2011.
Sentence 1: Joel Fuhrman (born December 2, 1953) is an American celebrity doctor who advocates what he calls a micronutrient-rich diet.
  Fact 1.0: Joel Fuhrman was born on December 2, 1953.
  Fact 1.1: Joel Fuhrman is American.
  Fact 1.2: Joel Fuhrman is a celebrity doctor.
  Fact 1.3: Joel Fuhrman advocates a diet that he calls a micronutrient-rich diet.

Retrieved items:
<item 1>
Eat to Live
Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss is a book written in 2003 by Dr. Joel Fuhrman, which he revised in 2011.

 The book offers a formula: health = nutrients / calories.
</item 1>

<item 2>
Diet food
Diet food (or dietetic food) refers to any food or beverage whose recipe is altered to reduce fat, carbohydrates, and/or sugar in order to make it part of a
weight loss program or diet.

 Such foods are usually intended to assist in weight loss or a change in body type, although bodybuilding supplements are designed to aid in gaining weight or
muscle.
</item 2>

<item 3>
Management of obesity
Management of obesity can include lifestyle changes, medications, or surgery.

 The main treatment for obesity consists of dieting and physical exercise.

 Diet programs may produce weight loss over the short term, but maintaining this weight loss is frequently difficult and often requires making exercise and a
lower calorie diet a permanent part of an individual's lifestyle.

 Success rates of long-term weight loss maintenance with lifestyle changes are low, ranging from 2 to 20%.

 Dietary and lifestyle changes are effective in limiting excessive weight gain in pregnancy and improve outcomes for both the mother and the child.

 The National Institutes of Health recommend a weight loss goal of 5% to 10% of the person's current weight over six months.
</item 3>

<item 4>
Very-low-calorie diet
Very low calorie diet (VLCD) is a diet with very or extremely low daily food energy consumption.

 It is defined as a diet of 800 kcal per day or less.

 VLCDs are formulated, nutritionally complete, liquid meals containing 800 kilocalories or less per day.

 VLCDs also contain the recommended daily requirements for vitamins, minerals, trace elements, fatty acids and protein.

 Carbohydrate may be entirely absent, or substituted for a portion of the protein; this choice has important metabolic effects.

 The VLCD products are usually a powder which is mixed with water or another low-food-energy liquid.

 The VLCD is prescribed on a case to case basis for rapid weight loss (about 1.5 to 2.5 kilograms or 3 to 5 pounds per week) in people with Body Mass Index of
30 and above.

 The health care provider can recommend the diet to a patient with a BMI between 27 and 30 if the medical complications the patient has due to being overweight
present serious health risks.

 It results in 4% more weight loss over the short term as compared to control.
</item 4>

<item 5>
Eat Smart, Move More, Weigh Less
Eat Smart, Move More, Weigh Less is a 15-week adult weight management program that uses strategies based on evidence for weight loss and/or weight maintenance.

 The Eat Smart Move More Weigh Less classes focus on the 12 evidence-based eating and physical activity behaviors for weight management.

 The program does not provide a prescriptive diet plan, it teaches small lifetime changes.

 The program teaches mindfulness as a strategy to become more aware of eating and physical activity.

 Eat Smart, Move More, Weigh Less includes methods for planning and tracking healthy eating and physical activity behaviors.

 In 2007, Eat Smart, Move More, Weigh Less curriculum was developed by a writing team from NC State University and NC Division of Public Health.

 A complete listing of authors and their respective affiliations can be found on the program website.

 The 15-lesson curriculum was peer reviewed by state and local nutrition and physical activity professionals, representatives from the medical community, and a
CDC project officer.
</item 5>

<item 6>
Diet in Hinduism
Diet in Hinduism varies with its diverse traditions.

 The ancient and medieval Hindu texts do not explicitly prohibit eating meat, but they do strongly recommend ahimsa—non-violence against all life forms
including animals.

 Many Hindus prefer a vegetarian or lacto-vegetarian lifestyle, and methods of food production that are in sync with nature, compassionate, and respectful of
other life forms as well as nature.
</item 6>

<item 7>
Joel Fuhrman
Joel Fuhrman (born December 2, 1953) is an American celebrity doctor who advocates what he calls a micronutrient-rich diet.
</item 7>

<item 8>
Scarsdale diet
The Scarsdale diet is a fad diet designed for weight loss created in the 1970s by Herman Tarnower, named for the town in New York where he practiced cardiology,
described in the book "The Complete Scarsdale Medical Diet plus Dr. Tarnower's Lifetime Keep-Slim Program", which Tarnower wrote with an author of self-help
books, Sam Sinclair Baker.

 The diet carries potential health risks and does not instill the kind of healthy eating habits required for sustainable weight loss.
</item 8>

<item 9>
Judy Mazel
Judy Mazel (December 20, 1943 – October 12, 2007) was the American weight loss advocate and author of the ""The Beverly Hills Diet"", which became a 1981 best
seller and a nationwide diet craze.

 Judy Mazel was a devout practicing Jew and was raised in a conservative Ashkenazi Jewish home.
</item 9>

<item 10>
Weight loss
Weight loss, in the context of medicine, health, or physical fitness, refers to a reduction of the total body mass, due to a mean loss of fluid, body fat or
adipose tissue or lean mass, namely bone mineral deposits, muscle, tendon, and other connective tissue.

 Weight loss can either occur unintentionally due to malnourishment or an underlying disease or arise from a conscious effort to improve an actual or perceived
overweight or obese state.

 "Unexplained" weight loss that is not caused by reduction in calorific intake or exercise is called cachexia and may be a symptom of a serious medical
condition.

 Intentional weight loss is commonly referred to as slimming.
</item 10>
=== END ITEM id=6eb165e6c7e7b20e ===

=== ITEM id=7298518e67266d89 ===
Gold sentences and facts:
Sentence 0: Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.
  Fact 0.0: Ted Kooshian was born October 8, 1961.
  Fact 0.1: Ted Kooshian is an American jazz pianist.
  Fact 0.2: Ted Kooshian is an American jazz keyboardist.
  Fact 0.3: Ted Kooshian is an American jazz composer.
  Fact 0.4: Ted Kooshian has performed with Aretha Franklin.
  Fact 0.5: Ted Kooshian has performed with Chuck Berry.
  Fact 0.6: Ted Kooshian has performed with Marvin Hamlisch.
  Fact 0.7: Ted Kooshian has performed with Sarah Brightman.
  Fact 0.8: Ted Kooshian has performed with Il Divo.
  Fact 0.9: Ted Kooshian has performed with Blood, Sweat, and Tears.
Sentence 1:  Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
  Fact 1.0: Marvin Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
Sentence 2:  This collection of all four is referred to as an "EGOT".
  Fact 2.0: The collection of all four awards (Emmy, Grammy, Oscar and Tony) is referred to as an 'EGOT'.

Retrieved items:
<item 1>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 1>

<item 2>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 2>

<item 3>
Marvin Hamlisch
Marvin Frederick Hamlisch (June 2, 1944August 6, 2012) was an American composer and conductor.

 Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.

 This collection of all four is referred to as an "EGOT".

 He is one of only two people (along with Richard Rodgers) to have won those four prizes and a Pulitzer Prize.
</item 3>

<item 4>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 4>

<item 5>
Lonesome Pine Fiddlers
This most talented family of musicians were the best, surpassed by none.

 Finally, on October 1, 2009, The Lonesome Pine Fiddlers got their due when they were inducted into the International Bluegrass Hall of Fame at the Ryman
Theater (the original home of the Grand Ole Opry).

 Bobby Osborne, Melvin Goins and Paul Williams were there to receive the bands award.

 In the crowd of a sold out theater was the son of Ezra Cline, Scotty Ireland Cline, who recalled being in that same theater as a child sitting on stage and
watching the Fiddlers play.

 (At the time, the Opry had bleachers for family just off stage).

 The final act of the evening at the IBMA Awards was the playing of "Pain in my Heart" by Osborne, Goins and Williams along with a Song from the Dillards, who
were also inducted the same evening.
</item 5>

<item 6>
List of awards and nominations received by Kim Tae-yeon
List of awards and nominations received by Kim Tae-yeon
</item 6>

<item 7>
List of James Tiptree Jr. Award winners
The following is a list of winners and shortlisted works of the James Tiptree Jr.

 Award, an annual literary prize for works of science fiction or fantasy that expand or explore one's understanding of gender.

 It was initiated in February 1991 by science fiction authors Pat Murphy and Karen Joy Fowler, and is awarded and discussed yearly at WisCon.
</item 7>

<item 8>
Profiles span musicians across jazz, rock, metal, electronic, gospel, and pop. Robert Glasper has three Grammys; Carlos Santana, ten Grammys and a landmark
Latin-rock fusion career. Daron Malakian (System of a Down), Gaahl (Gorgoroth), Roy Khan (Kamelot), Toshi (X Japan), and Rivermaya’s former guitarist Perfecto
de Castro are noted rock figures. Relient K’s Matt Thiessen, electronic artists Greg Davis and
</item 8>

<item 9>
Tatsuhiro Oshiro
Tatsuhiro Ōshiro (大城 立裕 , Ōshiro Tatsuhiro , born 19 September 1925) is a Japanese novelist and playwright.

 He was awarded the Akutagawa Prize in 1967 for his novella of the same year, "The Cocktail Party", which has been adapted for the stage and made into a film.

 Ōshiro has also been an innovator of the traditional Ryukyuan narrative dance form known as kumi odori.

 Having added twenty new pieces to the repertoire, Ōshiro is credited as having "single-handedly revived the genre that originated in the 18th century" by
incorporating Okinawa shibai (dramas in the Okinawan language) and distinctive rhythms to construct a fluid, hybrid cultural identity.

 His writings have been noted for making Okinawan culture and history accessible to Japanese readership, while his more popular works have been critically
praised for "offering an acute perspective on the psychological and moral implications of war and military occupation."
</item 9>

<item 10>
Premio Lo Nuestro 2003
In the Regional Mexican, Pilar Montenegro took three awards for Regional Mexican Song of the Year, Pop Song of the Year ("Quitame Ese Hombre"), and for Best
Regional Mexican Female Artist.

 In the tropical genre, Celia Cruz took home with four great awards of the night for Best Salsa Performance, Best Tropical Female Artist, Best Tropical Song of
the year and Tropical Album of the Year.

 At the night, the greatest performance was a medley of top Latin hits from the last 15 years, performed by the artists that made them famous, including Vikki
Carr, Son by Four, Los Ilegales, La Mafia, Luis Enrique, Wilfrido Vargas and Olga Tañón.

 There was a great tribute to Celia Cruz by the world-famous salsa group "Fania All-Stars", of which Cruz was a member during the 1970s, reunited for an
exclusive performance that rocked the house.
</item 10>
=== END ITEM id=7298518e67266d89 ===

=== ITEM id=fdb01d88821c42b9 ===
Gold sentences and facts:
Sentence 0: Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.
  Fact 0.0: Cars 2 is a film.
  Fact 0.1: Cars 2 was released in 2011.
  Fact 0.2: Cars 2 is American.
  Fact 0.3: Cars 2 is computer-animated.
  Fact 0.4: Cars 2 is an action comedy spy film.
  Fact 0.5: Cars 2 was produced by Pixar Animation Studios.
  Fact 0.6: Cars 2 was produced for Walt Disney Pictures.
Sentence 1: Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway musicals "Hamilton" and "In the Heights".
  Fact 1.0: Lin-Manuel Miranda was born January 16, 1980.
  Fact 1.1: Lin-Manuel Miranda is American.
  Fact 1.2: Lin-Manuel Miranda is a composer, lyricist, playwright, and actor.
  Fact 1.3: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical Hamilton.
  Fact 1.4: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical In the Heights.
Sentence 2:  He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".
  Fact 2.0: Lin-Manuel Miranda co-wrote the songs for Disney's Moana soundtrack.
  Fact 2.1: The Moana soundtrack is from 2016.
  Fact 2.2: Lin-Manuel Miranda is set to star in Disney's upcoming film Mary Poppins Returns.

Retrieved items:
<item 1>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 1>

<item 2>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 2>

<item 3>
Cars 2 (video game)
Cars 2 (also known as Cars 2: The Video Game) is a racing game developed by Avalanche Software.

 It is based on the film "Cars 2", the sequel to the 2006 film "Cars".

 Originally announced at E3 2011, the game was released by Disney Interactive on major platforms in North America on June 21, 2011, and in Australia 2 days
later.

 The game was released in Europe on July 22, 2011.

 The PlayStation 3 version of the game was reported to feature stereoscopic 3D gameplay.

 A Nintendo 3DS version was released on November 1, 2011 (the same day the film was released on DVD and Blu-ray).

 A PlayStation Portable version was released in North America on November 8, 2011.

 It was also released in Europe on November 4, 2011.

 It was the last Pixar game to be released for the PlayStation Portable.
</item 3>

<item 4>
Cabinet Battle
"Cabinet Battle #1", "Cabinet Battle #2", and "Cabinet Battle #3" are songs written for Act II of the musical "Hamilton", based on the life of Alexander
Hamilton, which premiered on Broadway in 2015.

 Lin-Manuel Miranda wrote both the music and lyrics to the song.
</item 4>

<item 5>
Wreck-It Ralph
Wreck-It Ralph is a 2012 American 3D computer-animated fantasy-comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 52nd Disney animated feature film.

 The film was directed by Rich Moore, who has directed episodes of "The Simpsons" and "Futurama", and the screenplay was written by Phil Johnston and Jennifer
Lee from a story by Moore, Johnston, and Jim Reardon.

 John Lasseter served as the executive producer.

 The film features the voices of John C. Reilly, Sarah Silverman, Jack McBrayer, and Jane Lynch.

 The film tells the story of the eponymous arcade game villain who rebels against his role and dreams of becoming a hero.

 He travels between games in the arcade and ultimately must eliminate a dire threat that could affect the entire arcade and one that Ralph himself
unintentionally started.
</item 5>

<item 6>
Cars 2
Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the sequel to 2006's "Cars", and features the voices of Owen Wilson, Larry the Cable Guy, Michael Caine, Emily Mortimer, John Turturro, and Eddie Izzard.

 In the film, race car Lightning McQueen and tow truck Mater head to Japan and Europe to compete in the World Grand Prix, but Mater becomes sidetracked with
international espionage.

 The film was directed by John Lasseter, written by Ben Queen, and produced by Denise Ream.
</item 6>

<item 7>
Moana (soundtrack)
Moana: Original Motion Picture Soundtrack is the soundtrack to the 2016 Disney animated film "Moana".

 The soundtrack was released by Walt Disney Records on November 19, 2016.

 It features songs written by Lin-Manuel Miranda, Mark Mancina and Opetaia Foa'i, with lyrics in English, Samoan, Tokelauan, and Tuvalu.

 The two-disc deluxe edition includes the score, which was composed by Mancina, as well as demos, outtakes and instrumental karaoke tracks.
</item 7>

<item 8>
Toy Story 3
Toy Story 3 is a 2010 American 3D computer-animated comedy-drama film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the third installment in Pixar's "Toy Story" series, and the sequel to 1999's "Toy Story 2".

 It was directed by Lee Unkrich, the editor of the first two films and the co-director of "Toy Story 2", written by Michael Arndt, while Unkrich wrote the story
along with John Lasseter and Andrew Stanton, respectively director and co-writer of the first two films.

 The plot focuses on the toys Woody, Buzz Lightyear, and their friends dealing with an uncertain future as their owner, Andy, prepares to leave for college.
</item 8>

<item 9>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 9>

<item 10>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 10>
=== END ITEM id=fdb01d88821c42b9 ===

=== ITEM id=4f22aca427d22159 ===
Gold sentences and facts:
Sentence 0: The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.
  Fact 0.0: The Long–Allen Bridge is a truss bridge.
  Fact 0.1: The Long–Allen Bridge is in Shreveport, Louisiana.
  Fact 0.2: The Long–Allen Bridge is named for Louisiana governor Huey Long.
  Fact 0.3: The Long–Allen Bridge is named for Louisiana governor Oscar K. Allen.
Sentence 1:  Opened in 1933, it carries U.S. Route 80 across the Red River.
  Fact 1.0: The Long–Allen Bridge opened in 1933.
  Fact 1.1: The Long–Allen Bridge carries U.S. Route 80 across the Red River.
Sentence 2: U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.
  Fact 2.0: U.S. Route 80 (US 80) is an east–west United States Numbered Highway.
  Fact 2.1: Much of U.S. Route 80 was once part of the early auto trail known as the Dixie Overland Highway.
Sentence 3:  The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
  Fact 3.0: The eastern terminus of U.S. Route 80 is in Tybee Island, Georgia.
  Fact 3.1: The eastern terminus of U.S. Route 80 is at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street.
  Fact 3.2: The eastern terminus of U.S. Route 80 is near the Atlantic Ocean.

Retrieved items:
<item 1>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 1>

<item 2>
The Long–Allen (Texas Street) Bridge, opened in 1933, carries US 80 over the Red River in Shreveport. Yuma’s 1915 Ocean To Ocean Bridge, Arizona’s earliest
through-truss and only Pennsylvania-truss bridge, carried US 80; restored for vehicles in 2002, it spans the Colorado River. US 80 once crossed the country but
now runs Dallas–Tybee Island. Florida’s Spanish River Boulevard ends at SR A1A;
</item 2>

<item 3>
Rocky Creek Bridge No. 01089
Rocky Creek Bridge No. 01089, also known as Ben Jones Bridge, is a concrete highway bridge spanning Rocky Creek along the Pacific Ocean coast of the U.S. state
of Oregon.

 The bridge crosses a gorge near the creek's mouth, about 2 mi south of Depoe Bay in Lincoln County.

 Built in 1927, the bridge originally carried U.S. Route 101 (the Oregon Coast Highway) over the creek.

 After a stretch of the highway was relocated to make it straighter, the abandoned piece of the old highway became Otter Crest Loop Road, a bypass west of the
new highway.
</item 3>

<item 4>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 4>

<item 5>
Low Bridge (song)
The popular song "Low Bridge, Everybody Down" was written in 1905 by Thomas S. Allen after Erie Canal barge traffic was converted from mule power to engine
power, raising the speed of traffic.

 Also known as "Fifteen Years on the Erie Canal", "Fifteen Miles on the Erie Canal", "Erie Canal Song", and "Mule Named Sal", the song memorializes the years
from 1825 to 1880 when the mule barges made boomtowns out of Utica, Rome, Syracuse, Rochester, and Buffalo, and transformed New York into the Empire State.

 The tune is sadly nostalgic.
</item 5>

<item 6>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 6>

<item 7>
Long–Allen Bridge (Shreveport)
The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.

 Opened in 1933, it carries U.S. Route 80 across the Red River.

 It is also known as the Texas Street Bridge.
</item 7>

<item 8>
Grand Central Parkway
The Grand Central Parkway (GCP) is a 14.61-mile (23.51 km) long parkway that stretches from the Triborough Bridge in New York City to Nassau County on Long
Island.

 At the Queens–Nassau border, it becomes the Northern State Parkway, which runs across the northern part of Long Island through Nassau County and into Suffolk
County, where it ends in Hauppauge.

 The westernmost stretch (from the Triborough Bridge to exit 4) also carries a short stretch of Interstate 278 (I-278).

 The parkway runs through Queens and passes the Cross Island Parkway, Long Island Expressway, LaGuardia Airport and Citi Field, home of the New York Mets.

 The parkway is designated New York State Route 907M (NY 907M), an unsigned reference route.

 Despite its name, the Grand Central Parkway was not named after Grand Central Terminal.
</item 8>

<item 9>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 9>

<item 10>
New York’s Erie Canal heritage includes Lock 52 (1849–53), canal buildings and archaeological sites; the 1836–42 Broad Street Aqueduct later became a road and
subway route. The 1905 song “Low Bridge” recalls mule-powered canal traffic. The 525-mile state canal system includes the Erie, Oswego, Cayuga–Seneca and
Champlain canals. The Genesee River and tributaries Black Creek and Oatka Creek drain western New
</item 10>
=== END ITEM id=4f22aca427d22159 ===

=== ITEM id=d66371329aff4d4e ===
Gold sentences and facts:
Sentence 0: Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.
  Fact 0.0: Psychopsis is abbreviated Psychp in horticultural trade.
  Fact 0.1: Psychopsis is a genus of 5 known species of orchids.
  Fact 0.2: Psychopsis orchids are native to northern South America.
  Fact 0.3: Psychopsis orchids are native to Central America.
  Fact 0.4: Psychopsis orchids are native to Trinidad.
Sentence 1: Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.
  Fact 1.0: Aubrieta is commonly known as Aubretia.
  Fact 1.1: Aubrieta is a genus of about 12 species of flowering plants.
  Fact 1.2: Aubrieta is in the cabbage family Brassicaceae.

Retrieved items:
<item 1>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 1>

<item 2>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 2>

<item 3>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 3>

<item 4>
Dregea
Dregea is a genus of vines in the Apocynaceae, first described as a genus with this name in 1838.

 It is native to Africa and southern Asia.
</item 4>

<item 5>
Aubrieta
Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.

 The genus is named after Claude Aubriet, a French flower-painter.

 It originates from southern Europe east to central Asia but is now a common garden escape throughout Europe.

 It is a low, spreading plant, hardy, evergreen and perennial, with small violet, pink or white flowers, and inhabits rocks and banks.

 It prefers light, well-drained soil, is tolerant of a wide pH range, and can grow in partial shade or full sun.
</item 5>

<item 6>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 6>

<item 7>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 7>

<item 8>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 8>

<item 9>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 9>

<item 10>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 10>
=== END ITEM id=d66371329aff4d4e ===
