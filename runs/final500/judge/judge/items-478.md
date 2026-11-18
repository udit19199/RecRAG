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

=== ITEM id=922aa5212f844281 ===
Gold sentences and facts:
Sentence 0: Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer, musician, director, actor, and comedian who holds dual British and American citizenship.
  Fact 0.0: Christopher Haden-Guest is the 5th Baron Haden-Guest.
  Fact 0.1: Christopher Haden-Guest was born on February 5, 1948.
  Fact 0.2: Christopher Haden-Guest is simply known as Christopher Guest.
  Fact 0.3: Christopher Guest is an English-American screenwriter.
  Fact 0.4: Christopher Guest is an English-American composer.
  Fact 0.5: Christopher Guest is an English-American musician.
  Fact 0.6: Christopher Guest is an English-American director.
  Fact 0.7: Christopher Guest is an English-American actor.
  Fact 0.8: Christopher Guest is an English-American comedian.
  Fact 0.9: Christopher Guest holds dual British and American citizenship.
Sentence 1:  Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.0: Christopher Guest is most widely known in Hollywood for having written his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.1: Christopher Guest is most widely known in Hollywood for having directed his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.2: Christopher Guest is most widely known in Hollywood for having starred in his series of comedy films shot in mock-documentary (mockumentary) style.
Sentence 2:  He also directed dark comedies, children's movies, and historical pictures.
  Fact 2.0: Christopher Guest directed dark comedies.
  Fact 2.1: Christopher Guest directed children's movies.
  Fact 2.2: Christopher Guest directed historical pictures.

Retrieved items:
<item 1>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 1>

<item 2>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 2>

<item 3>
Welcome, or No Trespassing
Welcome, or No Trespassing (Russian: Добро пожаловать, или Посторонним вход воспрещён , "Dobro pozhalovat, ili Postoronnim vkhod vospreshchyon " ) is a Soviet
movie by Elem Klimov made in 1964.

 It is a satirical comedy about the excessive restrictions that children face during their vacation in a Young Pioneer camp, imposed by their masters.

 Most of the actors are children, while the protagonist is the director Dynin, played by Yevgeniy Yevstigneyev.

 The film was selected to be screened in the Cannes Classics section of the 2015 Cannes Film Festival.
</item 3>

<item 4>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 4>

<item 5>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 5>

<item 6>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 6>

<item 7>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 7>

<item 8>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 8>

<item 9>
Farewell (1983 film)
Farewell (Russian: Прощание ) is a 1983 Soviet drama film based on Valentin Rasputin's novel "Farewell to Matyora" and directed by Elem Klimov.
</item 9>

<item 10>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 10>
=== END ITEM id=922aa5212f844281 ===

=== ITEM id=fbc46062623377fe ===
Gold sentences and facts:
Sentence 0: George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
  Fact 0.0: George Albert Smith Sr. was born on April 4, 1870.
  Fact 0.1: George Albert Smith Sr. died on April 4, 1951.
  Fact 0.2: George Albert Smith Sr. was an American religious leader.
  Fact 0.3: George Albert Smith Sr. served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
Sentence 1: Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.
  Fact 1.0: Wong Jing was born on 3 May 1955.
  Fact 1.1: Wong Jing is a Hong Kong film director.
  Fact 1.2: Wong Jing is a Hong Kong film producer.
  Fact 1.3: Wong Jing is a Hong Kong actor.
  Fact 1.4: Wong Jing is a Hong Kong presenter.
  Fact 1.5: Wong Jing is a Hong Kong screenwriter.

Retrieved items:
<item 1>
Feng Xingxi
Feng Xingxi (馮行襲) (died July 31, 910), courtesy name Zhengchen (正臣), formally Prince Zhongjing of Changle (長樂忠敬王), was a warlord late in the Chinese dynasty
Tang Dynasty who later became a subject of the succeeding Later Liang.

 He was tall and strong and known as "Green Face Feng" for his green birthmark on his face.
</item 1>

<item 2>
Albert Smith Medal
The Albert Smith Memorial Medal is the award given to the Man of the Match in the final of the Camanachd Cup, the blue riband trophy of the sport of shinty.

 It has been presented every year since 1972 by the Smith family of Fort William in honour of Albert Smith Sr. (Born in 1888, Lochuanagan, Fort Augustus.)

 The widow of his only son, Albert Smith Jr. currently presents the medal.

 Albert Sr.'s grandson, Victor Smith is a former player for Fort William Shinty Club but never won the medal despite featuring as a key player in several Fort
William wins.
</item 2>

<item 3>
Ba-Jia-Jiang
Ba Jia Jiang (八家將) originated from the Chinese folk beliefs and myths, and usually generally refers to eight members of the godly realm.

 The general understanding of the origin of Ba Jia Jiang is that it is derived from the existence of eight generals who performed exorcism of evil spirits for
the Wufu Emperor (五福大帝).

 These eight generals became revered as the gods of the underworld, and are represented as the bodyguards or attendants for the temples of the nether Gods such
as the Dongyue Emperor (東獄大帝), Yama (King of Hell, 閻羅王) and Cheng Huang (the City Gods, 城隍).

 Gradually Ba Jia Jiang evolved to appear also as bodyguards to Wang Ye (Royal Lord, 王爺) and Matsu (媽祖), and at many other temples.

 Later on, believers at those temples dressed up as Ba Jia Jiang in order to defend the Gods.
</item 3>

<item 4>
The Conman
The Conman is a 1998 Hong Kong action comedy film directed by Wong Jing and starring Andy Lau, Athena Chu and Nick Cheung.

 Despite the Chinese title, which translates as "Knight of Gamblers 1999", Andy Lau does not reprise his role as the "Knight of Gamblers" from the "God of
Gamblers" series, which was also directed by Wong Jing.

 The film was followed by a sequel "The Conmen in Vegas", which Lau and Cheung return with new cast members Natalis Chan, Kelly Lin, Meggie Yu and Alex Man.
</item 4>

<item 5>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 5>

<item 6>
George A. Smith
George Albert Smith (June 26, 1817 – September 1, 1875) (known throughout his life as George A. Smith) was an early leader in the Latter-day Saint movement.

 He served in the Quorum of the Twelve Apostles and as a member of the First Presidency of The Church of Jesus Christ of Latter-day Saints (LDS Church).
</item 6>

<item 7>
Chao-Li Chi
Chao-Li Chi (; April 5, 1927 – October 16, 2010) was a Shanxi-born actor and dancer who worked extensively in American television, including his best known role
as Chao-Li, the faithful majordomo and chauffeur of Jane Wyman's character in "Falcon Crest".

 Additionally, his film credits include "Big Trouble in Little China", "The Joy Luck Club", "The Nutty Professor", "Wedding Crashers" and "The Prestige".

 He was featured in the short film by Maya Deren, "Meditation on Violence", in 1948.
</item 7>

<item 8>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 8>

<item 9>
Mario Gaoa
Mario Gaoa (born 1971) is a New Zealand actor, writer and director, best known as a member of the Naked Samoans comedy group.

 He is of Samoan descent.

 As part of the group he has appeared in the film Sione's Wedding; provided the voices of Sione Tapili and God in the animated series Bro'Town, which he also
co-writes; and acted in various Naked Samoans comedic theatre performances.

 He has also appeared in the film Nightmare Man and briefly in the television series .
</item 9>

<item 10>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 10>
=== END ITEM id=fbc46062623377fe ===

=== ITEM id=b79710fcd2628fc8 ===
Gold sentences and facts:
Sentence 0: Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic series.
  Fact 0.0: Levni Yilmaz was born in 1973.
  Fact 0.1: Levni Yilmaz is a San Francisco based independent film maker.
  Fact 0.2: Levni Yilmaz is an artist.
  Fact 0.3: Levni Yilmaz is a publisher.
  Fact 0.4: Levni Yilmaz is best known for his 'Tales of Mere Existence' animated comic series.
Sentence 1: Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.
  Fact 1.0: Pamela B. Green is American.
  Fact 1.1: Pamela B. Green is a film producer.
  Fact 1.2: Pamela B. Green is a film director.
  Fact 1.3: Pamela B. Green is known for her work in feature film titles.
  Fact 1.4: Pamela B. Green is known for her work in motion graphics.

Retrieved items:
<item 1>
İsmet Ergün
İsmet Ergün (born 1950) is a Berlin artist and stage designer of Turkish provenance.

 Alongside her theatre work she has worked in the world of cinema as an art director and as a production designer.

 The short film which she directed, and for which she herself wrote the screenplay, was singled out for commendation at the 2007 Locarno Film Festival: it also
won the "Best German Film 2007" at the later the same year.
</item 1>

<item 2>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 2>

<item 3>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 3>

<item 4>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 4>

<item 5>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 5>

<item 6>
Pamela B. Green
Pamela B. Green is an American film producer and director known for her work in feature film titles and motion graphics.

 She is the director and producer of the documentary "Be Natural: The Untold Story of Alice Guy-Blache".
</item 6>

<item 7>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 7>

<item 8>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 8>

<item 9>
Pamela Gilbert
Pamela B. Gilbert (born October 3, 1958) is an American lawyer and has been a partner of the law firm Cuneo Gilbert & LaDuca, LLP since 2003, where she heads
the firm's lobbying practice.

 Gilbert is a noted consumer rights advocate who has testified before Congress over fifty times and made dozens of appearances in the national print and
electronic media.
</item 9>

<item 10>
Lev Yilmaz
Levni Yilmaz (born 1973) is a San Francisco based independent film maker, artist and publisher, best known for his “Tales of Mere Existence” animated comic
series.
</item 10>
=== END ITEM id=b79710fcd2628fc8 ===

=== ITEM id=f8d5f86c6e660c36 ===
Gold sentences and facts:
Sentence 0:  The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
  Fact 0.0: The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck.
  Fact 0.1: Dave Brubeck was also one of the company's investors.
  Fact 0.2: The label is known more for its recordings of comedian Lenny Bruce.
  Fact 0.3: The label is known more for its recordings of jazz musician Vince Guaraldi.
  Fact 0.4: The label is known more for its recordings of the rock band Creedence Clearwater Revival.
Sentence 1: Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
  Fact 1.0: Vincent Anthony Guaraldi was born on July 17, 1928.
  Fact 1.1: Vincent Anthony Guaraldi died on February 6, 1976.
  Fact 1.2: Vincent Anthony Guaraldi was born Vincent Anthony Dellaglio.
  Fact 1.3: Vincent Anthony Guaraldi was an American jazz pianist.
  Fact 1.4: Vincent Anthony Guaraldi was noted for his innovative compositions and arrangements.
  Fact 1.5: Vincent Anthony Guaraldi was noted for composing music for animated television adaptations of the 'Peanuts' comic strip.
  Fact 1.6: Vincent Anthony Guaraldi performed on piano as a member of Cal Tjader's late 50s ensemble.
  Fact 1.7: Vincent Anthony Guaraldi had a solo career.
  Fact 1.8: Vincent Anthony Guaraldi's solo career included the radio hit 'Cast Your Fate to the Wind'.

Retrieved items:
<item 1>
The material surveys jazz musicians, styles, songs, and albums. Vince Guaraldi composed music for Peanuts and had a hit with “Cast Your Fate to the Wind.” Count
Basie’s career and recordings are covered, including “One O’Clock Jump,” The Atomic Mr. Basie, and Beatles- and Bond-themed albums. Other profiles include Miles
Davis, Don Ellis, Thad Jones, Louis Smith, blues and boogie-woogie pianists, and jazz fusion’s late-196
</item 1>

<item 2>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 2>

<item 3>
John Fogerty (album)
John Fogerty is the second solo studio album by former Creedence Clearwater Revival vocalist/guitarist John Fogerty, released in 1975.

 It was released by Asylum Records in the United States and Fantasy Records internationally.

 As with the Creedence Clearwater Revival records, the album consists of a mix of originals and cover songs.

 Although the album is eponymously titled, Fogerty himself refers to it as "Old Shep"; Shep was the name of his dog, who appears on the cover with him.
</item 3>

<item 4>
The entries profile rock, punk, and metal albums, bands, and musicians. Fuel’s *Puppet Strings* (2014) reunited singer Brett Scallions but featured none of the
prior album’s recording personnel. Jizzy Pearl sang on Quiet Riot’s *10* and L.A. Guns’ *Shrinking Violet*. Other subjects include J. Geils Band, Green Day side
projects and collaborators, Glassjaw and Daryl Palumbo, Clearview 77, Sons of Abraham,
</item 4>

<item 5>
Vince Guaraldi
Vincent Anthony Guaraldi (July 17, 1928 – February 6, 1976), born Vincent Anthony Dellaglio, was an American jazz pianist noted for his innovative compositions
and arrangements and for composing music for animated television adaptations of the "Peanuts" comic strip, as well as his performances on piano as a member of
Cal Tjader's late 50s ensemble and his own solo career which included the radio hit "Cast Your Fate to the Wind".
</item 5>

<item 6>
Creedence Clearwater Revival’s 1968 debut launched a career later documented in compilations, including 1977’s *The Best of Creedence Clearwater Revival* and
2005’s *The Long Road Home*, which spans the band and John Fogerty’s solo work. A 2001 box set collects the band’s studio and live albums plus recordings under
earlier names. Fantasy Records, founded in 1949, became known for CCR. Drummer Doug Clifford and
</item 6>

<item 7>
The Long Road Home
The Long Road Home: The Ultimate John Fogerty/Creedence Collection is a compilation album by American roots rock singer-songwriter John Fogerty, released on
November 1, 2005, by Fantasy Records.

 It compiles songs from Fogerty's solo career and his band Creedence Clearwater Revival.

 The title refers to Fogerty's return to Fantasy Records, after a lengthy stint with Warner Bros.

 Records and a brief stint with DreamWorks Records.
</item 7>

<item 8>
The entries cover diverse music milestones: Marc Ribot’s 2005 Albert Ayler tribute, *Spiritual Unity*; comic duo Collins & Harlan, whose 1916 recording helped
introduce “jas,” precursor to “jazz”; and reggae singer Cedric Myton, a Royal Rasses and Congos member whose voice features on *Humanity* and *Heart of the
Congos*. They also describe early Dean Martin, Chico Hamilton, Shelly Manne, and Maynard
</item 8>

<item 9>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 9>

<item 10>
Fantasy Records
Fantasy Records is an American record company and label founded by brothers Max and Sol Weiss in 1949.

 The early years of the company were dedicated to issuing recordings by jazz pianist Dave Brubeck, who was also one of its investors, but the label is known
more for its recordings of comedian Lenny Bruce, jazz musician Vince Guaraldi, and the rock band Creedence Clearwater Revival.
</item 10>
=== END ITEM id=f8d5f86c6e660c36 ===

=== ITEM id=1d451006f4098f1c ===
Gold sentences and facts:
Sentence 0:  He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and the Highest Distinction of the Argentine Air Force.
  Fact 0.0: He was awarded the highest national military decoration, the Argentine Nation to the Heroic Valour in Combat Cross.
  Fact 0.1: He was awarded the Argentine Congressional Medal.
  Fact 0.2: He was awarded the Highest Distinction of the Argentine Air Force.
Sentence 1:  In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
  Fact 1.0: In 2010, the entity referred to here had 14,600 military personnel.
  Fact 1.1: In 2010, the entity referred to here had 6,900 civilian personnel.

Retrieved items:
<item 1>
South America air forces maneuvers
The South American air forces performs several joint aerial combat training exercises.

 Among the more important are Cruzex ( Portuguese: "Exercício Cruzeiro do Sul" ) which is hosted by the Brazilian Air Force, Salitre ( English: Saltpeter )
hosted by the Chilean Air Force and Ceibo ( English: Erythrina crista-galli flower ) hosted by the Argentine Air Force.

 The goal is to train together in order to respond to a crisis or integrate into United Nations peacekeeping operations as a unified team; the United States Air
Force and the Royal Canadian Air Force have also participated in some of these exercises in recent years.
</item 1>

<item 2>
Servicio de Inteligencia de la Fuerza Aérea
Servicio de Inteligencia de la Fuerza Aérea ("Air Force Intelligence Service", SIFA) is the intelligence agency of the Argentine Air Force.

 It is part of J-2.

 Its director is Commodore García.
</item 2>

<item 3>
Hector Fautario
Hector Luis Fautario was a retired Argentine Air Force general.

 He was the General Commander of the Argentine Air Force from 1973 to 1975.

 Shortly prior to the 1976 Argentine coup d'état, he was the government's only remaining loyal officer.

 He had drawn harsh criticism towards himself from the Army and Navy due to his vehement opposition to their repressive plans, and for his refusal to mobilize
the Air Force against the guerrillas' strongholds in the north.

 He was General Jorge Rafael Videla's last obstacle on the way to power.
</item 3>

<item 4>
Filiberto Ojeda Ríos
Filiberto Ojeda Ríos (April 26, 1933 – September 23, 2005) was the commander-in-chief ("Responsable General") of the Boricua Popular Army ("Ejército Popular
Boricua, a.k.a., Los Macheteros").

 According to an unsigned article in the "Los Angeles Times", Los Macheteros was “a terrorist group seeking Puerto Rico's independence.”

 The group campaigned for, and supported, the independence of Puerto Rico from the United States.

 In 2001, FBI Director Louis J. Freeh claimed the group was linked to acts of terrorism, but Ronald Fernandez, scholar of Puerto Rican history, suggests such
labeling was an act of political convenience by the United States Government, intended to "shift the blame for any attacks on U.S. policy or personnel from us
to them".

 Ríos was also a founder of the FALN.

 In a 1983 New York Times article, Robert McFadden described the FALN as a Puerto Rican terrorist organization responsible for bombings during the 1970s and
early 1980s "in the name of Puerto Rican independence".
</item 4>

<item 5>
Argentine Air Force
The Argentine Air Force (Spanish: Fuerza Aérea Argentina , or simply FAA) is the national aviation branch of the Armed Forces of the Argentine Republic.

 In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
</item 5>

<item 6>
Teodoro Waldner
Teodoro Guillermo Waldner (born 1927) is a retired Argentine Air Force commander who served as Air Force Chief of the General Staff from 1983 to 1985.
</item 6>

<item 7>
Pablo Carballo
Commodore Pablo Marcos Rafael Carballo (b. 11 December 1947 Buenos Aires) is a retired member of the Argentine Air Force - the "Fuerza Aérea Argentina" (FAA) -
who fought in the 1982 Falklands War ( Spanish: "Guerra de las Malvinas" ) where he participated in actions that led to the sinking of three Royal Navy ships.

 He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and
the Highest Distinction of the Argentine Air Force.
</item 7>

<item 8>
Cicaré CK.1
The Cicaré CK.1 (originally, the CH.III Colibrí) was a light helicopter developed in Argentina in the 1970s.

 It was a small, single-rotor aircraft of pod-and-boom configuration with a fully enclosed bubble canopy that could seat three people side-by-side.

 Cicaré's previous helicopter designs had attracted the attention of the Argentine Air Force, which in 1974 contracted him to develop a light helicopter for
training and also marketed for agricultural use.

 A prototype, registered "LV-X62" flew in September 1976, and the Air Force placed an order for five pre-production machines.

 However, development was terminated at this point.
</item 8>

<item 9>
Cuerpo de Fuerzas Especiales
The Mexican Cuerpo de Fuerzas Especiales (Special Forces Corps) is a special forces unit of the Mexican Army.

 Formerly the "GAFE" (Grupo Aeromóvil de Fuerzas Especiales | Special-Forces Airmobile Group), the SF Corps has six battalions; one is the "Fuerza especial de
reaccion", a quick-response unit, and one is assigned to the Paratroopers Rifle Brigade; the motto of the SF Corps is "Todo por México" (Everything for Mexico).

 Within the SF Corps, there are regular, intermediate, and veteran -service troops.

 The regular-service soldiers usually operate as light infantry.

 The intermediate-service soldiers (lieutenants and captains) usually are instructors.

 The veteran-service soldiers of the Grupo Aeromóvil de Fuerzas Especiales del Alto Mando (GAFE High Command) handle Black-Ops missions.

 Also known as the COIFE, the Special Forces Corps of the Mexican Army is equivalent to the U.S. Army Special Forces.
</item 9>

<item 10>
Argentine Air Force Mobile Field Hospital
The Argentine Air Force Mobile Field Hospital (Spanish: "Hospital Reubicable de la Fuerza Aérea" ) is a field hospital operated by the Argentine Air Force.

 It is one of three health centers of its kind worldwide.
</item 10>
=== END ITEM id=1d451006f4098f1c ===

=== ITEM id=78e97c1bd98effdf ===
Gold sentences and facts:
Sentence 0: Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.
  Fact 0.0: Mulled wine is a beverage.
  Fact 0.1: Mulled wine is usually made with red wine.
  Fact 0.2: Mulled wine is usually made with various mulling spices.
  Fact 0.3: Mulled wine is sometimes made with raisins.
Sentence 1:  It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.
  Fact 1.0: Mulled wine is served hot or warm.
  Fact 1.1: Mulled wine is alcoholic.
  Fact 1.2: There are non-alcoholic versions of mulled wine.
Sentence 2: Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
  Fact 2.0: Blue Lagoon is a popular summer cocktail.
  Fact 2.1: Blue Lagoon features blue Curaçao.

Retrieved items:
<item 1>
Feuerzangenbowle
Feuerzangenbowle (] ) is a traditional German alcoholic drink for which a rum-soaked sugarloaf is set on fire and drips into mulled wine.

 It is often part of a Christmas or New Year's Eve tradition.

 The name translates literally to "fire-tongs punch".
</item 1>

<item 2>
Mulled wine
Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.

 It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.

 It is a traditional drink during winter, especially around Christmas.
</item 2>

<item 3>
Mulling spices
Mulling spices is a spice mixture used in drink recipes.

 The spices are usually used to spice hot apple cider, mulled wine, wassail, and other drinks (such as juices) during the autumn or winter.

 A "mulled" drink is a drink which has been prepared with these spices (usually through heating the drink in a pot with mulling spices and then straining).

 Mulling spices may also be added to the brewing process to make spiced beer.
</item 3>

<item 4>
Blue Lagoon (cocktail)
Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
</item 4>

<item 5>
Oz and Hugh Drink to Christmas
Oz and Hugh Drink to Christmas is a BBC television programme in which wine personality Oz Clarke and comedian Hugh Dennis travel through Britain to sample a
wide array of seasonal Christmas beverages, including whisky, winter ales, mulled wine, wassail, sloe gin, Buck's Fizz, Port wine and Sherry.

 Upon its 20 December 2009 broadcast on BBC Two, it had a viewership of approximately 2.4 million with an audience share of 9%.
</item 5>

<item 6>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 6>

<item 7>
The Blue Lagoon (novel)
The Blue Lagoon is a romance novel written by Henry De Vere Stacpoole and was first published by T. Fisher Unwin in 1908.

 It is the first novel of the "Blue Lagoon" trilogy, which also includes "The Garden of God" (1923) and "The Gates of Morning" (1925).

 The novel has inspired several film adaptations, most notably "The Blue Lagoon" starring Brooke Shields as Emmeline and Christopher Atkins as Richard ("Dicky"
in the book).
</item 7>

<item 8>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 8>

<item 9>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 9>

<item 10>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 10>
=== END ITEM id=78e97c1bd98effdf ===

=== ITEM id=3490b9b476025c3d ===
Gold sentences and facts:
Sentence 0: "Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".
  Fact 0.0: 'Collateral Damage' is the eleventh episode of the third season of the crime-thriller television series 'Millennium'.
  Fact 0.1: 'Millennium' is a crime-thriller television series.
  Fact 0.2: 'Collateral Damage' is in the third season of 'Millennium'.
Sentence 1:  "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.
  Fact 1.0: 'Collateral Damage' featured a guest appearance by O'Quinn.
  Fact 1.1: 'Collateral Damage' featured a guest appearance by Jacinda Barrett.
  Fact 1.2: 'Collateral Damage' featured a guest appearance by James Marsters.
  Fact 1.3: 'Collateral Damage' featured a guest appearance by radio host Art Bell.
  Fact 1.4: Art Bell appeared as himself in 'Collateral Damage'.
Sentence 2:  Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
  Fact 2.0: Art Bell's radio show 'Coast to Coast AM' was among Perry's inspirations for the script.
Sentence 3: Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.
  Fact 3.0: Coast to Coast AM is an American late-night radio talk show.
  Fact 3.1: Coast to Coast AM deals with a variety of topics.
Sentence 4:  The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
  Fact 4.0: Coast to Coast AM now airs seven nights a week.
  Fact 4.1: Coast to Coast AM airs from 1:00 a.m. to 5:00 a.m. Eastern Time Zone.

Retrieved items:
<item 1>
Studio 60’s 2006 pilot, written by Aaron Sorkin and directed by Thomas Schlamme, introduces a chaotic fictional sketch show. The Office’s “Two Weeks” (2009)
follows Michael’s planned paper-company startup and Pam’s copier struggle; its 2013 finale closes the series as episodes 200–201. Two and a Half Men ran from
2003 to 2015. Charlie Sheen’s final episode, “That Darn Priest,” preceded his
</item 1>

<item 2>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 2>

<item 3>
Pilot (Studio 60 on the Sunset Strip)
"Pilot" is the first episode of the television series "Studio 60 on the Sunset Strip".

 The episode was first aired in the United States on the NBC network on September 18, 2006.

 Written by series creator Aaron Sorkin, and directed by executive producer Thomas Schlamme, the episode introduces the chaotic behind-the-scenes depiction of a
fictional "Saturday Night Live" type show also called "Studio 60 on the Sunset Strip".
</item 3>

<item 4>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 4>

<item 5>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 5>

<item 6>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 6>

<item 7>
Collateral Damage (Millennium)
"Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".

 It premiered on the Fox network on January 22, 1999.

 The episode was written by Michael R. Perry, and directed by Thomas J. Wright.

 "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.

 Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
</item 7>

<item 8>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 8>

<item 9>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 9>

<item 10>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 10>
=== END ITEM id=3490b9b476025c3d ===

=== ITEM id=b77efbbe9fc4f594 ===
Gold sentences and facts:
Sentence 0: Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.
  Fact 0.0: Terri Kathleen Nunn was born June 26, 1961.
  Fact 0.1: Terri Kathleen Nunn is an American singer.
  Fact 0.2: Terri Kathleen Nunn is an American actress.
Sentence 1: Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.0: Sonya Scarlet was born 2 April 1980.
  Fact 1.1: Sonya Scarlet is the singer of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.2: Sonya Scarlet is the lyricist of the Italian extreme gothic metal band Theatres des Vampires.

Retrieved items:
<item 1>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 1>

<item 2>
Sonya Yoncheva
Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
</item 2>

<item 3>
Nancy Spungen
Nancy Laura Spungen (February 27, 1958 – October 12, 1978) was the American girlfriend of Sex Pistols bassist Sid Vicious and a figure of the 1970s punk rock
scene.

 Spungen's life and death have been the subject of controversy among music historians and fans of the Sex Pistols.
</item 3>

<item 4>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 4>

<item 5>
Sonya Scarlet
Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
</item 5>

<item 6>
Terri Nunn
Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.

 She is best known as the lead vocalist of the new wave/synthpop band Berlin.
</item 6>

<item 7>
Briana Corrigan
Briana Corrigan (born 30 May 1965) is a Northern Irish singer.

 She was the first female singer for The Beautiful South, from 1988 to 1992.
</item 7>

<item 8>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 8>

<item 9>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 9>

<item 10>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 10>
=== END ITEM id=b77efbbe9fc4f594 ===

=== ITEM id=819f2213f55ac551 ===
Gold sentences and facts:
Sentence 0: Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.
  Fact 0.0: Hou Hsiao-hsien was born on 8 April 1947.
  Fact 0.1: Hou Hsiao-hsien is a Taiwanese film director.
  Fact 0.2: Hou Hsiao-hsien is a Taiwanese screenwriter.
  Fact 0.3: Hou Hsiao-hsien is a Taiwanese producer.
  Fact 0.4: Hou Hsiao-hsien is a Taiwanese actor.
Sentence 1:  He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.
  Fact 1.0: Hou Hsiao-hsien is a leading figure in world cinema.
  Fact 1.1: Hou Hsiao-hsien is a leading figure in Taiwan's New Wave cinema movement.
  Fact 1.2: Hou Hsiao-hsien is an auteur.
Sentence 2:  He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in 2015 for "The Assassin" (2015).
  Fact 2.0: Hou Hsiao-hsien won the Golden Lion at the Venice Film Festival in 1989.
  Fact 2.1: Hou Hsiao-hsien won the Golden Lion for his film "A City of Sadness" (1989).
  Fact 2.2: Hou Hsiao-hsien won the Best Director award at the Cannes Film Festival in 2015.
  Fact 2.3: Hou Hsiao-hsien won the Best Director award at Cannes for his film "The Assassin" (2015).
Sentence 3:  Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
  Fact 3.0: "The Puppetmaster" (1993) is a highly regarded work by Hou Hsiao-hsien.
  Fact 3.1: "Flowers of Shanghai" (1998) is a highly regarded work by Hou Hsiao-hsien.
Sentence 4: Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.
  Fact 4.0: Kenny Ortega (Kenneth John Ortega) was born on April 18, 1950.
  Fact 4.1: Kenny Ortega is an American producer.
  Fact 4.2: Kenny Ortega is an American director.
  Fact 4.3: Kenny Ortega is an American choreographer.
Sentence 5:  He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
  Fact 5.0: Kenny Ortega is best known for directing "Hocus Pocus".
  Fact 5.1: Kenny Ortega is best known for directing the "High School Musical" trilogy.
  Fact 5.2: Kenny Ortega is best known for directing "Descendants".
  Fact 5.3: Kenny Ortega is best known for directing Michael Jackson's "This Is It" concerts.

Retrieved items:
<item 1>
Kenny Ortega
Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.

 He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
</item 1>

<item 2>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 2>

<item 3>
Kenny Hotz
Kenneth Joel "Kenny" Hotz is a Canadian comedy writer, producer, director, actor and comedian.

 Hotz is a "South Park" consultant and writer, creator/star of the Comedy Central television show "Kenny vs. Spenny", creator and writer of the FX series
"Testees", and the creator/star of "Kenny Hotz's Triumph of the Will".

 He has received numerous awards for his television work— from the Academy of Canadian Cinema & Television the Gemini Awards and the Canadian Comedy Awards—and
is a multiple-time film festival award-winner.

 He is also an award-winning "Vice" contributor and Gulf war photo-journalist.
</item 3>

<item 4>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 4>

<item 5>
List of awards and nominations received by Kim Tae-yeon
List of awards and nominations received by Kim Tae-yeon
</item 5>

<item 6>
Hou Hsiao-hsien
Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.

 He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.

 He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in
2015 for "The Assassin" (2015).

 Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
</item 6>

<item 7>
The Assassin (2015 film)
The Assassin (; or: "The Assassin Niè Yǐnniáng") is a 2015 wuxia film directed by Taiwanese director Hou Hsiao-hsien.

 A Taiwan/China/Hong Kong co-production, it was an official selection in the main competition section at the 2015 Cannes Film Festival.

 At Cannes, Hou won the award for Best Director.

 It was released in China and Hong Kong on 27 August, and a day later in Taiwan on 28 August 2015.

 It was selected as the Taiwanese entry for the Best Foreign Language Film at the 88th Academy Awards but it was not nominated.
</item 7>

<item 8>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 8>

<item 9>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 9>

<item 10>
Academy Award for Best Film Editing
This process is essentially the reverse of that of the British Academy of Film and Television Arts (BAFTA); nominations for the BAFTA Award for Best Editing are
done by a general ballot of Academy voters, and the winner is selected by members of the editing chapter.
</item 10>
=== END ITEM id=819f2213f55ac551 ===

=== ITEM id=c85fd9d553daf81f ===
Gold sentences and facts:
Sentence 0: Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
  Fact 0.0: Race to Nowhere is a 2009 documentary film.
  Fact 0.1: Race to Nowhere was written by Maimone Attia.
  Fact 0.2: Race to Nowhere was directed by Vicki Abeles.
  Fact 0.3: Race to Nowhere was directed by Jessica Congdon.
Sentence 1: SlamNation is a documentary film by director Paul Devlin.
  Fact 1.0: SlamNation is a documentary film.
  Fact 1.1: SlamNation was directed by Paul Devlin.
Sentence 2:  The film follows the National Poetry Slam in Portland, Oregon.
  Fact 2.0: SlamNation follows the National Poetry Slam.
  Fact 2.1: The National Poetry Slam is in Portland, Oregon.

Retrieved items:
<item 1>
Bukowski (1973 film)
Bukowski is a 1973 documentary film produced by Taylor Hackford and directed by Richard Davies that follows Los Angeles poet Charles Bukowski to San Francisco
for a poetry reading.

 The full 60-minute documentary begins with footage of Bukowski in his Los Angeles home and neighborhood as he discusses his history as a postal worker as well
as his approach to and perspective on poetry.

 The film then shows him flying with Linda King to San Francisco for the poetry reading followed by interactions with attendees after the show.

 One night the window of his room is broken during a fight between some guests and then a fight between Charles and Linda causes her to leave.

 Interviews follow with Liza and Linda about their relationship with Charles.

 Bukowski is shown betting at the track and explaining his betting strategy.
</item 1>

<item 2>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 2>

<item 3>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 3>

<item 4>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 4>

<item 5>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 5>

<item 6>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 6>

<item 7>
Alice Walker: Beauty in Truth
Alice Walker: Beauty in Truth is a documentary film directed by Pratibha Parmar, made by Kali Films production company.

 The film follows the life of the Pulitzer Prize-winning author, poet and activist Alice Walker.

 Shooting began in May 2011.

 It was aired on the BBC on Monday July 8, 2013, and on PBS on February 7, 2014.
</item 7>

<item 8>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 8>

<item 9>
Race to Nowhere
Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
</item 9>

<item 10>
SlamNation
SlamNation is a documentary film by director Paul Devlin.

 The film follows the National Poetry Slam in Portland, Oregon.
</item 10>
=== END ITEM id=c85fd9d553daf81f ===

=== ITEM id=b68554d64ee54983 ===
Gold sentences and facts:
Sentence 0: Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also in Halland.
  Fact 0.0: Spettekaka is also called spettkaka.
  Fact 0.1: In native Scanian, spettekaka is called "spiddekaga".
  Fact 0.2: Spettekaka is a local dessert.
  Fact 0.3: Spettekaka is a local dessert of the southern parts of Sweden.
  Fact 0.4: Spettekaka is chiefly found in the province of Scania (Skåne).
  Fact 0.5: Spettekaka is also found in Halland.
Sentence 1: Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.
  Fact 1.0: Scanian is a closely related group of South Swedish dialects.
  Fact 1.1: Scanian dialects are spoken in the province of Scania.
  Fact 1.2: Scanian dialects are spoken in southern Sweden.
Sentence 2:  Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.0: Scanian formed part of the old Scandinavian dialect continuum.
  Fact 2.1: Most historical linguists consider Scanian to be an East Danish dialect group.
  Fact 2.2: Standard Swedish has influenced the Scanian dialects in the region in the modern era.
  Fact 2.3: The Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.
  Fact 2.4: The Scanian dialects have been treated as South Swedish due to the modern-era influence from Standard Swedish in the region.
  Fact 2.5: The Scanian dialects have been treated as South Swedish because traditional dialectology in the Scandinavian countries normally has not considered isoglosses that cut across state borders.

Retrieved items:
<item 1>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 1>

<item 2>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 2>

<item 3>
Scanian dialect
Scanian (   ) is a closely related group of South Swedish dialects spoken in the province of Scania in southern Sweden.

 Scanian formed part of the old Scandinavian dialect continuum and are by most historical linguists considered to be an East Danish dialect group, but due to
the modern-era influence from Standard Swedish in the region and because traditional dialectology in the Scandinavian countries normally has not considered
isoglosses that cut across state borders, the Scanian dialects have normally been treated as a South Swedish dialect group in Swedish dialect research.

 However, many of the early Scandinavian linguists, including Adolf Noreen and G. Sjöstedt, classified it as "South Scandinavian", and some linguists, such as
Elias Wessén, also considered Old Scanian a separate language, classified apart from both Old Danish and Old Swedish.
</item 3>

<item 4>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 4>

<item 5>
The Saga of Gosta Berling
The Saga of Gosta Berling (Swedish: Gösta Berlings saga ) is a 1924 Swedish romantic drama film directed by Mauritz Stiller and released by AB Svensk
Filmindustri, starring Lars Hanson, Gerda Lundequist and Greta Garbo in her native break-out role on film.

 The film is based on the 1891 debut novel of the same name by the Swedish author and Nobel Prize winner Selma Lagerlöf.

 It is also known as Gosta Berling's Saga, The Story of Gosta Berling and The Atonement of Gosta Berling.
</item 5>

<item 6>
Hakata dialect
Hakata dialect (博多弁 , Hakata-ben ) is a Japanese dialect spoken in Fukuoka city.

 Hakata dialect originated in Hakata commercial district, while a related Fukuoka dialect (福岡弁 , Fukuoka-ben ) was spoken in the central district.

 Hakata dialect has spread throughout the city and its suburbs.

 Most of Japanese regard Hakata dialect as the dialect typical of Fukuoka, so it is sometimes called "Fukuoka-ben".
</item 6>

<item 7>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 7>

<item 8>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 8>

<item 9>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 9>

<item 10>
Scoliopus
Scoliopus, or fetid adderstongue, is a genus of plant within the Liliaceae family consisting of two species, "Scoliopus bigelovii" and "S. hallii."

 Both are found in deep shaded forests, primarily in the coastal counties of the western United States from central California to northern Oregon.

 The name ""Scoliopus"" derives from the Greek words "skolios" and "pous", meaning curved foot, a reference to the shape of the pedicel.

 Taxonomists believe that "Scoliopus" is closely related to "Calochortus", "Prosartes", "Streptopus" and "Tricyrtis", which all have creeping rhizomes as well
as styles that divide at the tip.
</item 10>
=== END ITEM id=b68554d64ee54983 ===

=== ITEM id=26b735a246658695 ===
Gold sentences and facts:
Sentence 0: Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and Randall Counties, Texas, United States.
  Fact 0.0: Rick Husband Amarillo International Airport has the IATA code AMA.
  Fact 0.1: Rick Husband Amarillo International Airport has the ICAO code KAMA.
  Fact 0.2: Rick Husband Amarillo International Airport has the FAA LID AMA.
  Fact 0.3: Rick Husband Amarillo International Airport is a public airport.
  Fact 0.4: Rick Husband Amarillo International Airport is six miles (10 km) east of downtown Amarillo.
  Fact 0.5: Rick Husband Amarillo International Airport is in Potter and Randall Counties, Texas, United States.
Sentence 1: Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County, North Dakota.
  Fact 1.0: Grand Forks International Airport has the IATA code GFK.
  Fact 1.1: Grand Forks International Airport has the ICAO code KGFK.
  Fact 1.2: Grand Forks International Airport has the FAA LID GFK.
  Fact 1.3: Grand Forks International Airport is a public airport.
  Fact 1.4: Grand Forks International Airport is five miles (8 km) northwest of Grand Forks.
  Fact 1.5: Grand Forks International Airport is in Grand Forks County, North Dakota.

Retrieved items:
<item 1>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 1>

<item 2>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 2>

<item 3>
Grand Forks International Airport
Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County,
North Dakota.

 GFK has no scheduled passenger flights out of the country but has an "international" title (like many other airports) because it has customs service for
arrivals from Canada and other countries.
</item 3>

<item 4>
Brainerd Lakes Regional Airport is a city- and county-owned, general-aviation airport near Brainerd, Minnesota, with one commercial airline. Grand Sky,
established in 2015 at Grand Forks Air Force Base, is a 217-acre UAS business park with runway access. Grand Forks Municipal Airport closed after World War II
and became urban land; Grand Forks International Airport serves the area and offers customs but no scheduled international flights. Nearby Emerado had 414
residents in
</item 4>

<item 5>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 5>

<item 6>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 6>

<item 7>
Emerado, North Dakota
Emerado is a city in Grand Forks County, North Dakota, United States located near Grand Forks Air Force Base.

 It is part of the "Grand Forks, ND-MN Metropolitan Statistical Area" or "Greater Grand Forks".

 The population was 414 at the 2010 census.

 Grand Forks Air Force Base is located near Emerado.
</item 7>

<item 8>
Downtown Grand Forks
Downtown Grand Forks is the original commercial center of Grand Forks, North Dakota.

 Located on the western bank of the Red River of the North, the downtown neighborhood is situated near the fork of the Red River and the Red Lake River.

 While downtown is no longer the dominant commercial area of the Greater Grand Forks community, it remains the historic center of Grand Forks.

 An 80.4 acre portion was listed on the National Register of Historic Places in 2005, as Downtown Grand Forks Historic District.

 Today, downtown Grand Forks is home to many offices, stores, restaurants, and bars.
</item 8>

<item 9>
John D. Odegard School of Aerospace Sciences
The John D. Odegard School of Aerospace Sciences (UND Aerospace) is a multidisciplinary college within the University of North Dakota (UND) in Grand Forks,
North Dakota.

 The school was formed in 1968.

 The majority of the school's fleet of over 120 aircraft is based at nearby Grand Forks International Airport and is the largest fleet of civilian flight
training aircraft in North America.

 UND Aerospace also operates flight training centers in Crookston, Minnesota, and Phoenix, Arizona.

 Today, the school has many aerospace-related programs including commercial aviation (fixed wing and rotorcraft), air traffic control, airport management, Space
Studies, Computer Science, Atmospheric Sciences, and Earth System Science & Policy.

 Currently, the school has over 500 faculty and 1,900 students making it the second largest of UND’s degree-granting colleges.

 The present dean of the school is Dr. Paul Lindseth.
</item 9>

<item 10>
Grand Forks International
The Grand Forks International (GFI) is an annual international invitational baseball tournament hosted at James Donaldson Park in Grand Forks, British Columbia.

 The GFI is the largest invitational baseball tournament in Canada and is a large part of the culture in Grand Forks.

 The tournament is also notable for being run entirely by volunteers.
</item 10>
=== END ITEM id=26b735a246658695 ===
