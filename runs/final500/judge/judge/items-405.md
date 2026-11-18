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

=== ITEM id=a7f699fe5e5ee931 ===
Gold sentences and facts:
Sentence 0: Operalia, The World Opera Competition is an annual international competition for young opera singers.
  Fact 0.0: Operalia, The World Opera Competition is an annual international competition.
  Fact 0.1: Operalia is a competition for young opera singers.
Sentence 1:  Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti, Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
  Fact 1.0: Operalia was founded in 1993.
  Fact 1.1: Operalia was founded by Plácido Domingo.
  Fact 1.2: Operalia has helped launch the careers of several important artists.
  Fact 1.3: Operalia has helped launch the career of Joseph Calleja.
  Fact 1.4: Operalia has helped launch the career of Giuseppe Filianoti.
  Fact 1.5: Operalia has helped launch the career of Rolando Villazón.
  Fact 1.6: Operalia has helped launch the career of José Cura.
  Fact 1.7: Operalia has helped launch the career of Joyce DiDonato.
  Fact 1.8: Operalia has helped launch the career of Elizabeth Futral.
  Fact 1.9: Operalia has helped launch the career of Inva Mula.
  Fact 1.10: Operalia has helped launch the career of Ana María Martínez.
  Fact 1.11: Operalia has helped launch the career of Sonya Yoncheva.
Sentence 2: Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
  Fact 2.0: Sonya Yoncheva was born on 25 December 1981.
  Fact 2.1: Sonya Yoncheva’s Bulgarian name is Соня Йончева.
  Fact 2.2: Sonya Yoncheva is a Bulgarian operatic soprano.

Retrieved items:
<item 1>
Andrea Carroll (soprano)
Andrea Carroll is an American soprano who has had an active international career in concerts and operas since 2012.

 A finalist in the 2015 Operalia, The World Opera Competition, she is particularly known for her performances with the Houston Grand Opera and the Vienna State
Opera.
</item 1>

<item 2>
Rita Fornia
Rita Fornia (17 July 1878 – 27 October 1922) was an American opera singer.

 She began her career in Europe at the beginning of the 20th century singing coloratura soprano roles.

 Early on in her career her voice darkened and dropped slightly causing her to focus more within the mezzo-soprano repertoire while still singing some soprano
roles.

 She joined the Metropolitan Opera company in 1907 where she performed regularly in mostly supporting roles until her retirement in 1922.

 She is best remembered today for portraying the role of the Abbess in the original production of Puccini's "Suor Angelica" in 1918.
</item 2>

<item 3>
Ao Li
Ao Li (born 8 February 1988) is a Chinese operatic bass-baritone and voice teacher who is particularly known for his performances at the San Francisco Opera.

 In 2013 he won first prize in the Operalia, The World Opera Competition and in 2014 he won the Metropolitan Opera National Council Auditions.
</item 3>

<item 4>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 4>

<item 5>
Rachel Willis-Sørensen
Rachel Willis-Sørensen (born 1984) is an American operatic soprano.

 She was a winner of the Metropolitan Opera National Council Auditions in 2010.

 In 2014, she won first prize, the Birgit Nilsson Prize, and the Zarzuela Prize of the Operalia competition.

 Other awards include first place in the 2009 Eleanor McCollum Competition for Young Singers in Houston and the 2010 Sara Tucker Grant from the Richard Tucker
Music Foundation.
</item 5>

<item 6>
Maria Katzarava
Maria Katzarava (born 1984) is an opera singer from Mexico who achieved international fame by winning first place in the Operalia competition in the opera and
zarzuela categories.

 Prior to this, she came to the attention of notable Mexican tenor Ramón Vargas, who helped her get a scholarship, and then won Mexico's Carlo Morelli national
singing competition in 2005, which earned her a debut at the Palacio de Bellas Artes.

 Since winning the Operalia competition, Katzarava has performed in Europe, where she currently lives, Asia and the Americas.

 She currently trains with soprano Mirella Freni.
</item 6>

<item 7>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 7>

<item 8>
Guanqun Yu
Guanqun Yu (; born 1982) is a Chinese soprano who has sung in opera houses and concert halls internationally.

 In 2008 she won the Belvedere International Singing Competition and in 2012 she placed 2nd in the Operalia, The World Opera Competition.

 She is particularly known for portraying heroines in the operas of Wolfgang Amadeus Mozart, Giacomo Puccini, and Giuseppe Verdi.
</item 8>

<item 9>
Jane Rhodes
Jane Marie Andrée Rhodes (March 13, 1929 – May 7, 2011) was a French opera singer whose voice encompassed both the soprano and high mezzo-soprano ranges.

 Her most celebrated role was Carmen, which she sang in the opera's first ever staging at the Palais Garnier.

 She also created the roles of Isadora in Marcel Landowski's "Le Fou" and Maguelone in Georges Delerue's "Le Chevalier de Neige" and sang Renata in the first
recorded performance of Prokofiev's "The Fiery Angel".

 Admired for both her voice and her glamorous stage presence, she was nicknamed the "Bardot of the Opéra".

 She was married to conductor Roberto Benzi.
</item 9>

<item 10>
Norine Burgess
Norine Burgess is a Canadian singer.

 She is a graduate of the University of Calgary and the University of Toronto’s Opera School, mezzo-soprano.

 She received additional training as a member of the Canadian Opera Company (COC) Ensemble where she appeared in Electra, Suor Angelica, Lulu and Der
Rosenkavalier.

 Ms. Burgess also performed Le Nozze di Figaro(Cherubino), Ariadne auf Naxos (Dryad), La Traviata(Flora) and Die Zauberflöte (Second Lady) with the COC.

 Additional operatic successes include Le Nozze di Figaro (Cherubino) and Carmen (Mercedes) with the Vancouver Opera, Fenena in Nabucco with Manitoba Opera,
Albert Herring (Nancy) with the Calgary Opera and many appearances with the Edmonton Opera.
</item 10>
=== END ITEM id=a7f699fe5e5ee931 ===

=== ITEM id=9f5f80a45db6fab5 ===
Gold sentences and facts:
Sentence 0: Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and "".
  Fact 0.0: Steve Rash is an American film director.
  Fact 0.1: Steve Rash is an American producer.
  Fact 0.2: Steve Rash is best known for directing The Buddy Holly Story.
  Fact 0.3: Steve Rash is best known for directing Can't Buy Me Love.
  Fact 0.4: Steve Rash is best known for directing Queens Logic.
  Fact 0.5: Steve Rash is best known for directing two further films whose titles are blank in the source text.
Sentence 1: Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.
  Fact 1.0: Can't Buy Me Love is a 1987 American film.
  Fact 1.1: Can't Buy Me Love is a teen romantic comedy feature film.
  Fact 1.2: Can't Buy Me Love stars Patrick Dempsey.
  Fact 1.3: Can't Buy Me Love stars Amanda Peterson.
  Fact 1.4: The story of Can't Buy Me Love is about a nerd at a high school in Tucson, Arizona.
  Fact 1.5: In Can't Buy Me Love, the nerd gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

Retrieved items:
<item 1>
Bring It On: In It to Win It
Bring It On: In It to Win It is a 2007 teen film directed by Steve Rash and starring Ashley Benson, Cassie Scerbo and Michael Copon.
</item 1>

<item 2>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
</item 2>

<item 3>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 3>

<item 4>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 4>

<item 5>
The Cool Ones
The Cool Ones (aka "Cool, Baby Cool") is a 1967 film starring Roddy McDowall and directed by Gene Nelson.

 Mrs. Miller performs in a cameo role, and the film features cameo performances by bands the Leaves and the Bantams, as well as a cameo appearance by Glen
Campbell.
</item 5>

<item 6>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 6>

<item 7>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 7>

<item 8>
Steve Rash
Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and
"".
</item 8>

<item 9>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 9>

<item 10>
Can't Buy Me Love (film)
Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in
Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

 The film was directed by Steve Rash and takes its title from a Beatles song with the same title.
</item 10>
=== END ITEM id=9f5f80a45db6fab5 ===

=== ITEM id=01ea33301325264a ===
Gold sentences and facts:
Sentence 0:  It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that position for three weeks.
  Fact 0.0: The song was released as the lead single from the fourth studio album of its act, "Elephant."
  Fact 0.1: The song was released in March 2003.
  Fact 0.2: The song reached number one on the Modern Rock Tracks.
  Fact 0.3: The song maintained its number one position on the Modern Rock Tracks for three weeks.
Sentence 1:  Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album charts.
  Fact 1.0: The album "Elephant" was released on April 1, 2003.
  Fact 1.1: The album "Elephant" was released on V2 Records.
  Fact 1.2: The album "Elephant" garnered near unanimous critical acclaim.
  Fact 1.3: The album "Elephant" garnered commercial success.
  Fact 1.4: The album "Elephant" received a nomination for Album of the Year at the 46th Grammy Awards in 2004.
  Fact 1.5: The album "Elephant" won Best Alternative Music Album at the 46th Grammy Awards in 2004.
  Fact 1.6: The album "Elephant" peaked at No. 6 on the US "Billboard" charts.
  Fact 1.7: The album "Elephant" topped the UK album charts.

Retrieved items:
<item 1>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 1>

<item 2>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 2>

<item 3>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 3>

<item 4>
Seven Nation Army
"Seven Nation Army" (also stylized as "7 Nation Army") is a song by American rock duo The White Stripes.

 It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that
position for three weeks.

 It also became the third best-performing song of the decade on the same chart.

 It was well received commercially as well, and won the Grammy Award for Best Rock Song.
</item 4>

<item 5>
Late Night Tales: The Flaming Lips
Late Night Tales: The Flaming Lips is a compilation album compiled by the members of The Flaming Lips, featuring songs by various artists.

 The album was released on March 7, 2005 and it features one new Flaming Lips recording, a cover of the White Stripes' "Seven Nation Army".
</item 5>

<item 6>
The Boys Are Back (The Oak Ridge Boys album)
The Boys Are Back is the thirtieth studio album of country music group The Oak Ridge Boys.

 It was released in 2009 under the Spring Hill Music Group label.

 The album marked the group's return to secular country music after releasing gospel albums since 1992.

 The track "Seven Nation Army," a cover of The White Stripes's 2003 song from their album "Elephant", was released as the album's first single.
</item 6>

<item 7>
Meg White
Megan Martha "Meg" White (born December 10, 1974) is an American drummer and occasional singer known for her work with Jack White in the Detroit rock duo The
White Stripes.

 On an impulse, she played on Jack's drums in 1997.

 The two decided to form a band and began performing two months later, calling themselves The White Stripes because of their last name and Meg's affinity for
peppermint candy.

 The band quickly became a Detroit underground favorite, before reaching national, then international fame.

 White has been nominated for various awards as a part of the White Stripes, and has received four Grammy Awards.
</item 7>

<item 8>
Premio Lo Nuestro 2003
Premio Lo Nuestro 2003 was the 15th anniversary of the awards.

 the show was hosted by Mexican presenters Marco Antonio Regil and Adal Ramones.

 Juanes, Thalía, Marc Anthony, Pilar Montenegro, Sin Bandera, Banda el Recodo and other Latin music greats gave electrifying performances.

 In the show, there was 36 awards winners with 135 nominations.

 In Pop genre, Awards was given for : Album of the Year, Best Male Artist, Best Female Artist, Best Group or duo, Best New Artist and Song of the Year.

 In Rock Genre : Best Rock Album and Best rock Performer of the Year.

 In Tropical genre : Best Tropical Album of the Year, Best Tropical Male Artist, Best Tropical Female Artist, Best Tropical Group or Duo of the Year, Best
Tropical New Artist, Tropical Song of the Year, Best Merengue Performance, Best Salsa Performance and Best Traditional Performance.

 Juanes was the biggest winner of night, took home four awards Best Pop Male Artist, Best Music Video, Best Rock Performance, and Pop Song of the Year .
</item 8>

<item 9>
Grammy Award for Best Male Rap Solo Performance
The Grammy Award for Best Male Rap Solo Performance was an honor presented to male recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958, and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 9>

<item 10>
Grammy Award for Best Female Rap Solo Performance
The Grammy Award for Best Female Rap Solo Performance was an honor presented to female recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958 and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 10>
=== END ITEM id=01ea33301325264a ===

=== ITEM id=dc3123469dff085a ===
Gold sentences and facts:
Sentence 0: Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.
  Fact 0.0: Enrico Cocozza was born on 6 November 1921.
  Fact 0.1: Enrico Cocozza died on 27 December 1997.
  Fact 0.2: Enrico Cocozza was a Scottish filmmaker.
  Fact 0.3: Enrico Cocozza won many film awards during the 1940s and 1950s.
Sentence 1: ", which was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.0: A film was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.1: The film whose title is quoted in this fragment was nominated for the Academy Award for Best Foreign Language Film.

Retrieved items:
<item 1>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 1>

<item 2>
Hysterical Blindness (film)
Hysterical Blindness is a TV movie made for HBO, directed by Mira Nair and starring Gena Rowlands, Uma Thurman, Juliette Lewis and Ben Gazzara.

 The movie premiered on HBO on August 21, 2002 to good reviews.

 In 2003, Uma Thurman won a Golden Globe Award for her portrayal of Debby Miller.

 Ben Gazzara and Gena Rowlands also won Best Supporting Actor/Actress awards for their performances as Virginia Miller and Nick Piccolo at the 2003 Emmy Awards.
</item 2>

<item 3>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 3>

<item 4>
Enrico Cocozza
Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.

 His often surreal films were mainly filmed in and around the town of Wishaw in Scotland, where his family owned the popular Belhaven Cafe.

 These include "Chick's Day" (1950), a prize winner at the 1951 Scottish Amateur Film Festival, "The Living Ghost" (1957), and "Glasgow's Docklands" (1959).

 Illness later forced him to give up making films and he spent most of his working life teaching at the University of Strathclyde in Glasgow.

 In 2001 he was the subject of the Channel 4 documentary "Artery: the Story of Enrico Cocozza".
</item 4>

<item 5>
Kathryn Drysdale
Kathryn Drysdale (born December 1981) is an English actress known for her role as Grace Shelley in The West End show "The Ruling Class" opposite James Mcavoy,
as Taylor in the movie "St Trinian's", Rhoda Swartz in Mira Nair's Vanity Fair, and Louise Brooks in the BBC comedy series "Two Pints of Lager and a Packet of
Crisps" from 2001 until her departure from the show in 2009.

 She has also starred in the fourth series of the ITV sitcom "Benidorm" and as Brownwell in William Boyd's award-winning "Any Human Heart" opposite Jim
Broadbent.

 She previously worked with Broadbent on Mira Nair's feature adaptation of William Thackery's "Vanity Fair" starring Reese Witherspoon.

 She played the role of mixed-race heiress Rhoda Swartz.

 She is also known for her role as Lizzie in the drama series "Tripping Over" by Mike Bullen.

 She appeared in "Doctor Who" as Bliss in the episode "Love & Monsters".

 She currently plays Meghan Markle in season 2 of "The Windsors".
</item 5>

<item 6>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 6>

<item 7>
Vanity Fair (2004 film)
Vanity Fair is a 2004 British-American historical drama film directed by Mira Nair and adapted from William Makepeace Thackeray's novel of the same name.

 The novel has been the subject of numerous television and film adaptations, and Nair's version made notable changes in the development of main character Becky
Sharp.
</item 7>

<item 8>
Nitish Roy
Nitish Roy is an Indian film art director, production designer, and costume designer in Hindi cinema and a Bengali Film Director, who is also known for his work
with art cinema directors, Shyam Benegal, Mrinal Sen and Govind Nihlani, Hindi mainstream cinema, directors like Rajkumar Santoshi, as well as international
directors like Mira Nair and Gurinder Chadha.
</item 8>

<item 9>
Salaam Bombay!
Salaam Bombay!

 is a 1988 Hindi film directed by Mira Nair, and screenwritten by her longtime creative collaborator, Sooni Taraporevala.

 The film chronicles the day-to-day life of children living on the streets of Bombay, India's biggest city.

 It won the National Film Award for Best Feature Film in Hindi, the National Board of Review Award for Top Foreign Film, the Golden Camera and Audience Awards
at the Cannes Film Festival, and three awards at the Montréal World Film Festival.

 The film was India's second film submission to be nominated for the Academy Award for Best Foreign Language Film.

 The film was among the list of "The Best 1,000 Movies Ever Made" by "The New York Times".
</item 9>

<item 10>
The Namesake (film)
The Namesake is a 2006 Indian-American drama film which was released in the United States on 9 March 2007, following screenings at film festivals in Toronto and
New York City.

 It was directed by Mira Nair and is based upon the novel of the same name by Jhumpa Lahiri, who appeared in the movie.

 Sooni Taraporevala wrote the screenplay.

 The film received positive reviews from American critics.

 The film stars Tabu, Irrfan Khan, Kal Penn and Sahira Nair.
</item 10>
=== END ITEM id=dc3123469dff085a ===

=== ITEM id=c6f1bac468893a87 ===
Gold sentences and facts:
Sentence 0: Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.
  Fact 0.0: Elizabeth Austin was born in 1958.
  Fact 0.1: Elizabeth Austin is an American writer.
  Fact 0.2: Elizabeth Austin lives in Oak Park, Illinois.
Sentence 1:  She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".
  Fact 1.0: Elizabeth Austin has also written for The Washington Monthly.
  Fact 1.1: Elizabeth Austin has also written for Ladies Home Journal.
  Fact 1.2: Elizabeth Austin has also written for Good Housekeeping.
  Fact 1.3: Elizabeth Austin has also written for Self.
  Fact 1.4: Elizabeth Austin has also written for Shape.
Sentence 2: Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight fitness) and became the number one women's fitness magazine.
  Fact 2.0: Shape is a monthly English language fitness magazine.
  Fact 2.1: Shape was started by Weider Publications in 1981.
  Fact 2.2: Shape was founded by Christine MacIntyre.
  Fact 2.3: Christine MacIntyre is a pioneer in women's free weight fitness.
  Fact 2.4: Shape became the number one women's fitness magazine.

Retrieved items:
<item 1>
Muscle &amp; Fitness (British magazine)
Muscle & Fitness is a British fitness magazine, which contains articles on strength and fitness training, diet plans and strategies for men and women, and tips
and advice.
</item 1>

<item 2>
Ingrid Romero
Ingrid Romero (born July 10, 1985) is an international fitness model, fitness competitor, and fitness coach.

 Training & Fitness Magazine called her "one of the most recognized faces in the fitness industry," and "one of the most accomplished female fitness stars
ever."
</item 2>

<item 3>
Anais Zanotti
Anais Zanotti (born January 8, 1986 in Saint-Tropez) is a French model and stunt woman of French and Italian ancestry who lives in Miami, Florida.

 She has been featured in magazines such as "Playboy", "FHM", "Maxim", "GQ", "Cosmopolitan", "Vogue", and "People", Men's Fitness magazine, Muscle and Fitness,
Women Fitness magazine and more.

 She has over 40 magazine covers.

 Zanotti is also a certified skydiver with over 1250 jumps.GS .
</item 3>

<item 4>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 4>

<item 5>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 5>

<item 6>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 6>

<item 7>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 7>

<item 8>
Computer Magazine
"Computer Magazine" is a free publication (in addition to their webcasts and other resources) sponsored by the nonprofit UTCP (United Technology and Computing
Professionals) organization, and as such charges no fees for the publication nor is influenced by advertising, so their reviews are relied on in the industry
and considered unbiased and thorough.

 "Computer Magazine" is one of the early large technology publications and resources available on the web still existent and thriving today and that has
remained independent.

 ComputerMagazine.com is a tech news and resources consolidator that publishes part of the site in a semi-time line/blogging format that is popular among their
wide following of subscriber and non-subscriber readers, allowing readers to respond and comment on various articles.

 Site contributors include many of the well known technology authors, experts and publication sources, content and articles are provided by major technology
syndicators and by external expert technology sources (such as "Computer World", "Information Week", "Network World", "Wired," "Time", etc.)
</item 8>

<item 9>
Elizabeth Austin (writer)
Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.

 Austin has lived in the Midwestern United States all her life.

 She began her journalism career at the now-defunct City News Bureau of Chicago and has written articles for national news magazines such as "Newsweek" and
"Time", in addition to a wide range of other national and regional publications, such as "People", and the "Chicago Tribune".

 She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".

 She has often written about medicine and health care.

 Austin also has been heard on WBEZ, the National Public Radio station in Chicago, as well as the BBC in the UK and Scotland.

 She has appeared on the Today Show and on ITV in Europe.
</item 9>

<item 10>
Wendy Austin
Dr Wendy Elizabeth Austin Hewitt, Wendy Austin is a BBC journalist and broadcaster in BBC Northern Ireland, and former presenter of BBC Radio Ulster’s current
affairs programme "Talkback".
</item 10>
=== END ITEM id=c6f1bac468893a87 ===

=== ITEM id=6807526e365fc28d ===
Gold sentences and facts:
Sentence 0:  It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts Mika Brzezinski and Willie Geist.
  Fact 0.0: The program described features former GOP representative Joe Scarborough.
  Fact 0.1: Joe Scarborough provides enterprise reporting on the program.
  Fact 0.2: Joe Scarborough provides discussion on the news of the day on the program.
  Fact 0.3: The program is in a panel format.
  Fact 0.4: Mika Brzezinski is a co-host of the program.
  Fact 0.5: Willie Geist is a co-host of the program.
Sentence 1: Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.
  Fact 1.0: Charles Joseph Scarborough was born on April 9, 1963.
  Fact 1.1: Charles Joseph Scarborough is an American cable news host.
  Fact 1.2: Charles Joseph Scarborough is an American talk radio host.

Retrieved items:
<item 1>
Ronn Owens
Ronn Owens (born Ronald Lowenstein on October 17, 1945, in New York City), is an American talk radio host.

 Owens was the sole remaining weekday talk radio host on KGO in San Francisco after KGO was reorganized as an all-news station in December 2011.
</item 1>

<item 2>
Mike Hogan (sportscaster)
Mike Hogan (born 1963 in Kingston, Ontario) is a Canadian sportscaster who is a sports talk radio host on TSN Radio 1050 in Toronto.

 He also serves as the play-by-play voice for the Toronto Argonauts of the Canadian Football League for that station's gameday broadcasts.

 Hogan was formerly a long time sports talk radio host on Toronto station CJCL (AM), branded "The Fan 590", until he was relieved of those duties on June 24,
2010.

 On 2017 Oscar Vanity fair party 1st live broadcast Mike Hogan was the podcast hosts.
</item 2>

<item 3>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 3>

<item 4>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 4>

<item 5>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 5>

<item 6>
Joe Scarborough
Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.

 He is currently the co-host of "Morning Joe" on MSNBC, and previously hosted "Scarborough Country" on the same channel.

 Scarborough was previously a lawyer and a politician, and served in the United States House of Representatives from 1995 to 2001 as a Republican from the 1st
district of Florida.
</item 6>

<item 7>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
</item 7>

<item 8>
Talk radio
Talk radio is a radio format containing discussion about topical issues and consisting entirely or almost entirely of original spoken word content rather than
outside music.

 Jason Culpepper is highly recognized as the father of talk radio.

 Jason a Jacksonville native rose to social prominence during the World War utilizing talk radio to warn french and american troops of German U boats.

 Most shows are regularly hosted by a single individual, and often feature interviews with a number of different guests.

 Talk radio typically includes an element of listener participation, usually by broadcasting live conversations between the host and listeners who "call in"
(usually via telephone) to the show.

 Listener contributions are usually screened by a show's producers in order to maximize audience interest and, in the case of commercial talk radio, attract
advertisers.

 Generally, the shows are organized into segments, each separated by a pause for advertisements; however, in public or non-commercial radio, music is sometimes
played in place of commercials to separate the program segments.

 Variations of talk radio include conservative talk, hot talk, liberal talk (increasingly known as progressive talk) and sports talk.
</item 8>

<item 9>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 9>

<item 10>
Sean Hannity
Sean Patrick Hannity (born December 30, 1961) is an American talk show host, author, and conservative political commentator.

 Hannity is the host of "The Sean Hannity Show", a nationally syndicated talk radio show.

 He also hosts a cable news show, "Hannity", on Fox News Channel.
</item 10>
=== END ITEM id=6807526e365fc28d ===

=== ITEM id=333ff329948ec02c ===
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
Yutmaru Sar
Yutmaru Sar is a mountain in the Hispar mountain range, a subrange of the Karakoram.

 At an elevation of 7283 m it is the 88th highest mountain in the world.

 Yutmaru Sar is located in Gilgit-Baltistan, Pakistan.

 It was first climbed in 1980.
</item 1>

<item 2>
The Crown (mountain)
The Crown, also known as Huang Guan Shan and sometimes Crown Peak, is a mountain in the Karakoram mountain range in China.

 It is located in the Xinjiang Uyghur Autonomous Region of China.

 Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
</item 2>

<item 3>
Kanjut Sar
Kanjut Sar (Urdu: کنجت سر‎ ) or Kunjudh Sar as pronounced in "Wakhi" is a mountain located in the Hispar Muztagh, a subrange of the Karakoram mountain range.

 Kunjudh Sar in wakhi language mean that which overlooks Kunjudh, or above Kunjudh, while Khujudh is the wakhi name for Lower Hunza.

 It is the 26th highest mountain on Earth and the 11th highest in Pakistan.
</item 3>

<item 4>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 4>

<item 5>
Momhil Sar
Momhil Sar, or Mumhail Sar as pronounced in Wakhi is at 7343 m above sea level, is the 64th highest mountain peak in the world.

 Mumhail Sar in Wakhi means the mountain that overlooks or is above Grandmother's cattle pen or paddock.

 It is situated in the Hispar Muztagh subrange of the Karakoram range, a few kilometres to the north-west of its parent peak Trivor.
</item 5>

<item 6>
Kezhen Peak
Kezhen Peak, also known as Karpogo Sar, is a mountain in the Karakoram mountain range.

 It is located in the Xinjiang Autonomous Region of China.
</item 6>

<item 7>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 7>

<item 8>
Passu Sar
Passu Sar (Urdu: ‎ ; or Passu Sar, Passu I) is a mountain peak in the Batura Muztagh, a sub-range of the Karakoram mountain range, located in the Gilgit
District of Gilgit-Baltistan, Pakistan, west of the Hunza Valley.

 It is the high point of the Passu massif, which also includes Passu Diar (or "Passu East", "Pasu II").

 The peak lies on the main ridge of the Batura Muztagh, about 7 km (4 mi) east of Batura Sar.
</item 8>

<item 9>
Yengisogat
The Yengisogat range (音苏盖提 ), also known as the Wesm Mountains, is a Chinese subrange of the Karakoram mountain range.

 It lies north of the Baltoro Muztagh, home of the eight-thousanders of the Karakoram.

 The highest peak is Huangguan Shan, or Crown Peak, 7,265 m (23,835 ft) (also sometimes given as 7,295 m/23,934 ft).
</item 9>

<item 10>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 10>
=== END ITEM id=333ff329948ec02c ===

=== ITEM id=b60117c41a176347 ===
Gold sentences and facts:
Sentence 0: Rivermaya is a Filipino rock band.
  Fact 0.0: Rivermaya is a Filipino rock band.
Sentence 1: Kodaline ( ) are an Irish band.
  Fact 1.0: Kodaline is an Irish band.

Retrieved items:
<item 1>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 1>

<item 2>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 2>

<item 3>
Rivermaya
Rivermaya is a Filipino rock band.

 Formed in 1994, it is one of several bands who spearheaded the 1990s Philippine alternative rock explosion.

 Rivermaya is currently composed of original members Mark Escueta and Nathan Azarcon, together with Mike Elgar and Ryan Peralta.

 Former original members include Rico Blanco, who had been the original songwriter of the band and vocalist Bamboo Mañalac, who later formed the band Bamboo and
later went on his solo career.

 Rivermaya is listed as the twentieth biggest-selling artists/act in the Philippines as of present.
</item 3>

<item 4>
Kodaline
Kodaline ( ) are an Irish band.

 Originally known as 21 Demands, in 2012 the band changed their name to Kodaline.
</item 4>

<item 5>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 5>

<item 6>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 6>

<item 7>
Diwa de Leon
Diwa de Leon is a composer, arranger and musician based in Manila, Philippines.

 He makes music mainly for television shows, films, and stage productions.

 His most notable work is his arrangement of GMA Network's "Survivor Philippines" theme music.

 His songs included in "Emir".

 He won the Cinema One Originals Best Musical Score award for his work on the film "Kolorete" and Best Music Score for "Kamera Obskura" at the 8th Cinemalaya
Independent Film Festival.

 He is also a founding member of the world music group Makiling.

 His first mainstream film project is Mamarazzi produced by Regal Films.
</item 7>

<item 8>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 8>

<item 9>
Honest (Kodaline song)
"Honest" is a song by Dublin-based alternative rock quartet Kodaline.

 The song was released on 1 February 2015 as the lead single from the band's second studio album, "Coming Up for Air" (2015).

 "Honest" became the band's third top 10 single in their home country, Ireland, following "High Hopes" (2013) and "Love Like This" (2013).
</item 9>

<item 10>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 10>
=== END ITEM id=b60117c41a176347 ===

=== ITEM id=de77bfe23b2f35df ===
Gold sentences and facts:
Sentence 0: Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist of the twentieth century".
  Fact 0.0: Erving Goffman was born on 11 June 1922.
  Fact 0.1: Erving Goffman died on 19 November 1982.
  Fact 0.2: Erving Goffman was a Canadian-American sociologist.
  Fact 0.3: Erving Goffman was a writer.
  Fact 0.4: Some consider Erving Goffman the most influential American sociologist of the twentieth century.
Sentence 1:  In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens, Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
  Fact 1.0: In 2007 Erving Goffman was listed by The Times Higher Education Guide.
  Fact 1.1: The Times Higher Education Guide listed Goffman as the sixth most-cited author in the humanities and social sciences in 2007.
  Fact 1.2: In that 2007 listing, Anthony Giddens, Pierre Bourdieu and Michel Foucault were ranked ahead of Goffman.
  Fact 1.3: In that 2007 listing, Goffman was ranked ahead of Jürgen Habermas.
Sentence 2: Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist, and literary critic.
  Fact 2.0: Paul-Michel Foucault was born on 15 October 1926.
  Fact 2.1: Paul-Michel Foucault died on 25 June 1984.
  Fact 2.2: Paul-Michel Foucault is generally known as Michel Foucault.
  Fact 2.3: Michel Foucault was a French philosopher.
  Fact 2.4: Michel Foucault was a historian of ideas.
  Fact 2.5: Michel Foucault was a social theorist.
  Fact 2.6: Michel Foucault was a literary critic.

Retrieved items:
<item 1>
Dramaturgy (sociology)
Dramaturgy is a sociological perspective commonly used in microsociological accounts of social interaction in everyday life.

 The term was first adapted into sociology from the theatre by Erving Goffman, who developed most of the related terminology and ideas in his 1959 book, "The
Presentation of Self in Everyday Life".

 Kenneth Burke, whom Goffman would later acknowledge as an influence, had earlier presented his notions of dramatism in 1945, which in turn derives from
Shakespeare.

 However, the fundamental difference between Burke's and Goffman's view is that Burke believed that life was in fact theatre, whereas Goffman viewed theatre as
a metaphor.

 If we imagine ourselves as directors observing what goes on in the theatre of everyday life, we are doing what Goffman called dramaturgical analysis, the study
of social interaction in terms of theatrical performance.
</item 1>

<item 2>
Ray Birdwhistell
Collaborations with others, including initially Margaret Mead and Gregory Bateson, and later, Erving Goffman and Dell Hymes had huge influence on his work.

 For example, the book he is best known for, "Kinesics and Context", "would not have appeared if it had not been envisaged by Erving Goffman" and he explicitly
stated "the paramount and sustaining influence upon my work has been that of anthropological linguistics", a tradition most directly represented at the
University of Pennsylvania by Hymes.
</item 2>

<item 3>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 3>

<item 4>
Erving Goffman
Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist
of the twentieth century".

 In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens,
Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
</item 4>

<item 5>
Identity Performance
In everyday interactions, the body serves as a critical site of identity performance.

 In conveying who we are to other people, we use our bodies to project information about ourselves.

 This is done through movement, clothes, speech, and facial expressions.

 What we put forward is our best effort at what we want to say about who we are.

 Yet while we intend to convey one impression, our performance is not always interpreted as we might expect.

 Through learning to make sense of others’ responses to our behavior, we can assess how well we have conveyed what we intended.

 We can then alter our performance accordingly.

 This process of performance, interpretation, and adjustment is what Erving Goffman calls impression management.

 Impression management is a part of a larger process where people seek to define a situation through their behavior.

 People seek to define social situations by using contextual cues from the environment around them.

 Social norms emerge from situational definitions, as people learn to read cues from the environment and the people present to understand what is appropriate
behavior.
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
Goffman (disambiguation)
Erving Goffman (1922–1982) was a Canadian sociologist
</item 7>

<item 8>
The Presentation of Self in Everyday Life
The Presentation of Self in Everyday Life is a 1956 sociology book by Erving Goffman, in which Goffman uses the imagery of the theatre in order to portray the
importance of human social interaction.

 Originally published in Scotland in 1956 and in the United States in 1959, it is Goffman’s first and most famous book, for which he received the American
Sociological Association’s MacIver award in 1961.
</item 8>

<item 9>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 9>

<item 10>
The entries survey literature, theatre, film, music, and curiosities: French Decadence favored artifice and excess; Camus staged Dostoyevsky’s *The Possessed*
in 1959; Algren won the 1950 National Book Award; Poe satirized Gothic fiction; Moorcock wrote Elric and science-fiction novels; Nin’s diaries and erotica
appeared partly posthumously. Wilde’s *Dorian Gray* faced censorship and condemnation. Verne’s
</item 10>
=== END ITEM id=de77bfe23b2f35df ===

=== ITEM id=0355d264bd9c2d4a ===
Gold sentences and facts:
Sentence 0: Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.
  Fact 0.0: Jeffrey Charles "Jeff" Ragsdale is an American author.
  Fact 0.1: Jeffrey Charles "Jeff" Ragsdale is a documentary filmmaker.
  Fact 0.2: Jeffrey Charles "Jeff" Ragsdale is an actor.
  Fact 0.3: Jeffrey Charles "Jeff" Ragsdale is a stand-up comedian.
Sentence 1: Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.
  Fact 1.0: Millard Webb was born on 6 December 1893.
  Fact 1.1: Millard Webb died on 21 April 1935.
  Fact 1.2: Millard Webb was an American screenwriter.
  Fact 1.3: Millard Webb was an American director.

Retrieved items:
<item 1>
Jeff, One Lonely Guy
Jeff, One Lonely Guy is a 2012 nonfiction book by Jeff Ragsdale.

 It was published on March 20, 2012 by New Harvest.

 Dave Eggers selected the book for inclusion in The Best American Nonrequired Reading 2012, and it was a GQ 2012 "Book of the Year".

 In 2014 Amitava Kumar included portions of "Jeff, One Lonely Guy" in his newly released book, "A Matter of Rats: A Short Biography of Patna".

 Kumar previously interviewed Ragsdale and wrote about him in "The New York Times".
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
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 3>

<item 4>
Jeff Ragsdale
Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.

 In 2011 he posted a flyer in New York City as a "social experiment", stating his phone number and asking people to call him, describing himself as "Jeff, one
lonely guy".

 He was overwhelmed with thousands of calls after photos of the flyer were posted on the internet.

 The experience led to his 2012 book "Jeff, One Lonely Guy", and indirectly to a 2013 pilot episode for a reality television show, "Being Noticed", and a
starring role in the 2014 documentary "Hotline".
</item 4>

<item 5>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 5>

<item 6>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 6>

<item 7>
Kim Edwards
Kim Edwards (born May 4, 1958) is an American author and educator.

 She was born in Killeen, Texas, grew up in the Finger Lakes region of upstate New York, and graduated from Colgate University and The University of Iowa, where
she earned an MFA in fiction and an MA in linguistics.

 She is the author of a story collection, "The Secrets of a Fire King", which was a finalist for the PEN/Hemingway Award; her stories have been published in
"The Paris Review, Story, Ploughshares, Zoetrope", and many other periodicals.

 She has received many awards for the short story as well, including a Pushcart Prize, the National Magazine Award, the Nelson Algren Award, and inclusion in
both "The Best American Short Stories" and the Symphony Space program ‘Selected Shorts.’

 She is the recipient of a Whiting Writers’ Award, as well as grants from the Pennsylvania and Kentucky Arts Councils, the Kentucky Foundation for Women, and
the National Endowment for the Arts.
</item 7>

<item 8>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 8>

<item 9>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 9>

<item 10>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 10>
=== END ITEM id=0355d264bd9c2d4a ===

=== ITEM id=ef2c480c86dfb287 ===
Gold sentences and facts:
Sentence 0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great Britain and is dedicated to Saint Chad of Mercia.
  Fact 0.0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham.
  Fact 0.1: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the province of the Catholic Church in Great Britain.
  Fact 0.2: The Metropolitan Cathedral Church and Basilica of Saint Chad is dedicated to Saint Chad of Mercia.
Sentence 1:  Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English Reformation and raised to cathedral status in 1852.
  Fact 1.0: St Chad's was built by Augustus Welby Pugin.
  Fact 1.1: St Chad's was substantially complete by 1841.
  Fact 1.2: St Chad's is one of the first four Catholic churches constructed after the English Reformation.
  Fact 1.3: St Chad's was raised to cathedral status in 1852.
Sentence 2: Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.
  Fact 2.0: Chelmsford Cathedral is in the city of Chelmsford, Essex, England.
  Fact 2.1: Chelmsford Cathedral is dedicated to St Mary the Virgin.
  Fact 2.2: Chelmsford Cathedral is dedicated to St Peter.
  Fact 2.3: Chelmsford Cathedral is dedicated to St Cedd.
Sentence 3:  It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
  Fact 3.0: Chelmsford Cathedral became a cathedral when the Anglican Diocese of Chelmsford was created in 1914.
  Fact 3.1: Chelmsford Cathedral is the seat of the Bishop of Chelmsford.

Retrieved items:
<item 1>
Ellis Gowing
Gowing was educated at Fort Street High School and the University of Sydney; and ordained in 1907.

 He began his career as a Curate at The Oaks, New South Wales, after which he served at Armidale Cathedral.

 Coming to England he was on the staff of St James the Less, Bethnal Green then domestic chaplain to the Bishop of Chelmsford from 1914 to 1917.

 He was vicar of Prittlewell from 1917 until his death; an honorary canon of Chelmsford Cathedral from 1921 to 1938 and Rural Dean of Canewdon from 1918 to
1938.

 He was given the Freedom of the County Borough of Southend-on-Sea in 1953.
</item 1>

<item 2>
Bishop's House, Birmingham
The Bishop's House in Birmingham, England was designed by Augustus Pugin as the residence of Thomas Walsh, the first Roman Catholic Bishop of Birmingham.

 It was situated opposite St Chad's Cathedral, on the corner of Bath Street and Weaman Street in Birmingham City Centre.
</item 2>

<item 3>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 3>

<item 4>
La Belle Iron Works
La Belle Iron Works, also known as La Belle Cut Nail Works, is a historic factory complex and national historic district located at Wheeling, Ohio County, West
Virginia.

 The district includes four contributing buildings; three Italianate style brick buildings dated to the founding of the company in 1852, and a tin plate mill
built 1894-1897.

 After 1902, the buildings were combined under a single roof, although the truss systems date to different periods achieving the configuration visible today.

 When listed in 1997, it was known as the "La Belle Cut Nail Plant, The Largest in the World, Wheeling Corrugating Company, A Division of Wheeling Pittsburgh
Steel Corporation."

 The cut nail machinery still in use by La Belle dates to 1852 and the 1860s.

 The machinery at La Belle along with the different processes were documented by the Historic American Buildings Survey team during the summer of 1990.
</item 4>

<item 5>
St Chad's Cathedral, Birmingham
The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great
Britain and is dedicated to Saint Chad of Mercia.

 Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English
Reformation and raised to cathedral status in 1852.

 It is one of only four minor basilicas in England (the others being Downside Abbey, the National Shrine of Our Lady at Walsingham and Corpus Christi Priory,
this last now disused).

 St Chad's is a Grade II* listed building.

 The cathedral is located in a public greenspace near St Chad's Queensway, in central Birmingham.

 s of 2014 the Archbishop was Bernard Longley and the Dean Canon Gerry Breen.
</item 5>

<item 6>
Dick Herrick
The Very Rev Richard William Herrick (3 December 1913 - 5 May 1981) was an eminent Anglican priest in the 20th century.

 He was educated at King Edward VI School Retford and Leeds University and was initially a civil servant.

 He was ordained after a period of study at the College of the Resurrection, Mirfield in 1940.

 he held curacies at Duston and Portsea, Portsmouth before being appointed Vicar of St Michael’s, Northampton in 1947, a post he held for a decade.

 He was then a Canon Residentiary of Chelmsford Cathedral until 1978 when he was appointed Provost of Chelmsford.

 He died in post.
</item 6>

<item 7>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 7>

<item 8>
Saint Michael the Archangel Church (Monroe, MI)
Saint Michael the Archangel Church (St. Michael Church) is located on the west side of the city of Monroe, Michigan along the River Raisin.

 It is home to 1,200 families and it is one of most important religious institutes in Monroe County.

 It was founded in the year 1852.

 The present building was built from 1866-1867.

 It is in the Archdiocese of Detroit.

 Its current priest is Rev. Phillip Ching.

 When the parish was first established, the mayor of Monroe palatial residence was remodeled as a temporary church.

 It was used as a church on the first floor and on the second it was used for the school.

 Later it would only be used as St. Michael School.

 Then in 1866 the cornerstone for the present church was laid.

 The large 187 foot steeple wasn't added until 1883.

 In 1874 the 3-story rectory was built east of the church.

 In 1918 the parish built the present building of St.
</item 8>

<item 9>
St Chad's College, Durham
St Chad's College is a recognised (independent) college of Durham University in England, founded in 1904 as an Anglican hall for the training of Church of
England clergy.

 The main part of the college is located on the Bailey, occupying nine historic buildings at the east end of Durham Cathedral.

 It neighbours Hatfield College to its north, while St John's College and St Cuthbert's Society are to its south.

 The college is named after St Chad of Mercia, a 7th-century bishop.
</item 9>

<item 10>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 10>
=== END ITEM id=ef2c480c86dfb287 ===
