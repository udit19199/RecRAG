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

=== ITEM id=9660e11a2fdcf616 ===
Gold sentences and facts:
Sentence 0:  Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest commercial success is the erotic thriller "Chloe" (2009).
  Fact 0.0: Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter".
  Fact 0.1: "The Sweet Hereafter" was released in 1997.
  Fact 0.2: Egoyan received two Academy Award nominations for "The Sweet Hereafter".
  Fact 0.3: Egoyan's biggest commercial success is the erotic thriller "Chloe".
  Fact 0.4: "Chloe" was released in 2009.
Sentence 1: Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".
  Fact 1.0: Chloe is a 2009 erotic thriller film.
  Fact 1.1: Chloe was directed by Atom Egoyan.
  Fact 1.2: Chloe is a remake of the 2003 French film "Nathalie...".

Retrieved items:
<item 1>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 1>

<item 2>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 2>

<item 3>
Various Positions (film)
Various Positions is a 2002 film by Vancouver, BC, lawyer and filmmaker Ori Kowarsky, starring Carly Pope and Tygh Runyan.

 "Various Positions" won the 2002 Prix de Montréal at the Montreal World Film Festival.

 Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any
affiliation with the work.
</item 3>

<item 4>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 4>

<item 5>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 5>

<item 6>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 6>

<item 7>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 7>

<item 8>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 8>

<item 9>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 9>

<item 10>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 10>
=== END ITEM id=9660e11a2fdcf616 ===

=== ITEM id=cc14bb7a9cb5e638 ===
Gold sentences and facts:
Sentence 0: Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.
  Fact 0.0: Sir Nicholas Robert Hytner was born on 7 May 1956.
  Fact 0.1: Sir Nicholas Robert Hytner is an English theatre director.
  Fact 0.2: Sir Nicholas Robert Hytner is an English film director.
  Fact 0.3: Sir Nicholas Robert Hytner is an English film producer.
Sentence 1: Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
  Fact 1.0: Maya Deren was born on April 29, 1917.
  Fact 1.1: Maya Deren died on October 13, 1961.
  Fact 1.2: Maya Deren was born Eleanora Derenkowskaia.
  Fact 1.3: The birth name Eleanora Derenkowskaia is given in Russian as Элеоно́ра Деренко́вская.
  Fact 1.4: Maya Deren was a Russian-American filmmaker.
  Fact 1.5: Maya Deren was one of the most important American experimental filmmakers in the 1940s and 1950s.
  Fact 1.6: Maya Deren was one of the most important American entrepreneurial promoters of the avant-garde in the 1940s and 1950s.
Sentence 2:  Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
  Fact 2.0: Maya Deren was also a choreographer.
  Fact 2.1: Maya Deren was also a dancer.
  Fact 2.2: Maya Deren was also a film theorist.
  Fact 2.3: Maya Deren was also a poet.
  Fact 2.4: Maya Deren was also a lecturer.
  Fact 2.5: Maya Deren was also a writer.
  Fact 2.6: Maya Deren was also a photographer.

Retrieved items:
<item 1>
Nicholas Mele
Nicholas Mele is an actor who has starred in many movies and on television.

 His first movie role was in the 1976 movie "The Ritz".

 Other movie roles include "Some Kind of Hero" (1982) and "Young Doctors in Love" (1982).

 His most well-known film role was in the 1988 hit horror movie "" as Dennis Johnson.

 A year later, he reprised his role in the hit sequel "".

 His most recent movie is the 2003 film "The Great Gabble".
</item 1>

<item 2>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 2>

<item 3>
Filmworks X: In the Mirror of Maya Deren
Filmworks X: In the Mirror of Maya Deren features a score for film by John Zorn.

 The album was released on Zorn's own label, Tzadik Records, in 2001 and contains music that Zorn wrote and recorded for the documentary film "In the Mirror of
Maya Deren" on the life and work of Maya Deren directed by Martina Kudlácek.
</item 3>

<item 4>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 4>

<item 5>
Maya Deren Award
The American Film Institute Award for Independent Film and Video Artists, subtitled and generally known as the Maya Deren Award, was an award presented to
filmmakers and video artists by the American Film Institute to honor independent filmmaking.

 Named for the avant-garde experimental film artist Maya Deren, it was given from 1986 through 1996.
</item 5>

<item 6>
Maya Deren
Maya Deren (April 29, 1917 – October 13, 1961), born Eleanora Derenkowskaia (Russian: Элеоно́ра Деренко́вская ), was a Russian-American filmmaker and one of the
most important American experimental filmmakers and entrepreneurial promoters of the avant-garde in the 1940s and 1950s.

 Deren was also a choreographer, dancer, film theorist, poet, lecturer, writer and photographer.
</item 6>

<item 7>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 7>

<item 8>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 8>

<item 9>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 9>

<item 10>
Mary Sweeney
Mary Sweeney is an American film producer, director, writer and film editor, who collaborated for 20 years with her husband, avant-garde American film director,
David Lynch.

 Sweeney worked with Lynch on several films and television series, most notably the original "Twin Peaks" series (1990), "Lost Highway" (1997), "The Straight
Story", (1999) and "Mulholland Drive" (2001).

 Sweeney is the Dino and Martha De Laurentiis Endowed Professor in the Writing Division of the School of Cinematic Arts at the University of Southern
California.

 She is the Chair of the Film Independent Board of Directors.
</item 10>
=== END ITEM id=cc14bb7a9cb5e638 ===

=== ITEM id=73839b8fe047e064 ===
Gold sentences and facts:
Sentence 0:  Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.0: Reed is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.1: Mike Pringle is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.2: Johnny Bright is one of the players most often mentioned as being the greatest running back in CFL history.
Sentence 1: Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.
  Fact 1.0: Johnny D. Bright was born June 11, 1930.
  Fact 1.1: Johnny D. Bright died December 14, 1983.
  Fact 1.2: Johnny D. Bright was a professional Canadian football player.
  Fact 1.3: Johnny D. Bright played in the Canadian Football League.
Sentence 2:  He played college football at Drake University.
  Fact 2.0: Johnny D. Bright played college football at Drake University.

Retrieved items:
<item 1>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 1>

<item 2>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 2>

<item 3>
Richard Harris (American football)
Richard Drew Harris (January 21, 1948 – July 26, 2011) was an American football defensive end who played seven seasons in the National Football League.

 He was and All-American in 1970 for Grambling and was drafted in the first round (5th overall pick) of the 1971 NFL Draft by the Philadelphia Eagles.

 He was named All-Rookie in 1971, playing defensive end.

 Harris spent seven seasons as a defensive end in the NFL, the first three with the Philadelphia Eagles, the next two with the Chicago Bears and the final two
with the Seattle Seahawks.
</item 3>

<item 4>
Skip Peete
Skip Peete (born January 30, 1963) is an American football coach who serves as the current running backs coach for the Los Angeles Rams of the National Football
League (NFL).

 Prior to the Rams, he coached running backs for the Chicago Bears, Dallas Cowboys, and Oakland Raiders.

 He also has experience in college, coaching running backs for two years at UCLA, wide receivers at Michigan State and Rutgers, and both running backs and wide
receivers at the University of Pittsburgh.
</item 4>

<item 5>
George Reed (Canadian football)
George Robert Reed, CM, SOM (born October 2, 1939), is a former American college football and Canadian Football League player.

 Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.

 In November 2006, Reed was voted one of the CFL's Top 50 players (#2) of the league's modern era by Canadian sports network TSN.
</item 5>

<item 6>
2,000-yard club
The 2,000-yard club is a group of seven National Football League (NFL) running backs that have rushed for 2,000 or more yards in a season.

 These seven rushing seasons rank as the highest single-season rushing totals in NFL history, and reaching the 2,000-yard mark is considered a significant
achievement for running backs.

 No running back has yet achieved this feat twice.

 The first 2,000-yard season was recorded in 1973 by Buffalo Bills running back O.J. Simpson.

 He is the only player to have surpassed 2,000 yards in a 14-game season, as all others occurred in 16-game seasons; he finished the season with 2,003 rushing
yards, averaging six yards per carry and an NFL-record 143.1 rushing yards per game.

 Los Angeles Rams running back Eric Dickerson, who had broken the single-season rookie rushing record in 1983, recorded the second 2,000-yard season in 1984.

 Dickerson rushed for 2,105 yards, the current NFL rushing record, and averaged 131.6 rushing yards per game.
</item 6>

<item 7>
Dolph Pulliam
Adolphus "Dolph" Pulliam (born March 14, 1946 in West Point, Mississippi) is a former American basketball player.

 He played collegiately at Drake University.

 He was named to Drake University's All-Century basketball team.

 He helped lead the Drake men's basketball team to the 1969 Final Four and a third-place finish.

 On 7 February 2009, Drake University retired Pulliam's #5 jersey that he wore for the Bulldogs.
</item 7>

<item 8>
Marc Mueller
Marc Mueller (born April 15, 1989) is the current running backs coach of the Calgary Stampeders of the Canadian Football League (CFL).

 In college, he played quarterback for the Regina Rams of the CIS.

 He signed as an undrafted free agent with the Edmonton Eskimos of the CFL on May 9, 2011 following the 2011 CFL Draft.

 He was released on June 20, 2011 in order to play his final year for the Rams.

 Mueller is the grandson of former CFL quarterback and coach Ron Lancaster.
</item 8>

<item 9>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 9>

<item 10>
Damon Allen
Damon Allen (born July 29, 1963) is a former professional quarterback who played in the Canadian Football League.

 He is currently second in all-time professional football passing yards and second in all-time CFL passing yards after he was surpassed for first place by the
Montréal Alouettes' Anthony Calvillo on October 10, 2011.

 He is pro-football's combined passing & rushing leader with a total of 84,301 yards.

 Allen retired as professional football's all-time leading passer with 72,381 passing yards after he surpassed Warren Moon's total of 70,553 yards (in both the
CFL and NFL combined) on September 4, 2006 in the annual Labour Day Classic.

 He also retired in third place in all-time CFL rushing yards with 11,920 yards, behind Mike Pringle and George Reed.

 The 2007 season marked Allen's twenty-third season in the CFL and he officially announced his retirement on May 28, 2008 at age 44.

 Allen is the younger brother of Pro Football Hall of Famer Marcus Allen.
</item 10>
=== END ITEM id=73839b8fe047e064 ===

=== ITEM id=7ccbd42f89d30432 ===
Gold sentences and facts:
Sentence 0: Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
  Fact 0.0: Gunnar Eric Nelson was born on September 20, 1967.
  Fact 0.1: Gunnar Eric Nelson is an American musician.
  Fact 0.2: Gunnar Eric Nelson is an American singer.
  Fact 0.3: Gunnar Eric Nelson is an American songwriter.
  Fact 0.4: Gunnar Eric Nelson is an international multi-platinum recording artist.
Sentence 1: Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the Machine.
  Fact 1.0: Florence Leontine Mary Welch was born on 28 August 1986.
  Fact 1.1: Florence Leontine Mary Welch is an English musician.
  Fact 1.2: Florence Leontine Mary Welch is an English singer.
  Fact 1.3: Florence Leontine Mary Welch is an English songwriter.
  Fact 1.4: Florence Leontine Mary Welch is best known as the vocalist of the indie rock band Florence and the Machine.

Retrieved items:
<item 1>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 1>

<item 2>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 2>

<item 3>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 3>

<item 4>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 4>

<item 5>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 5>

<item 6>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 6>

<item 7>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 7>

<item 8>
Gunnar Nelson (musician)
Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
</item 8>

<item 9>
Sweet Nothing
"Sweet Nothing" is a song by Scottish recording artist and record producer Calvin Harris from his third studio album, "18 Months" (2012).

 It features vocals from Florence and the Machine singer Florence Welch.

 Harris previously worked with the band in a remix of their single, "Spectrum" (2012).

 While recording "18 Months", Harris expressed interest in working with Welch.

 He mentioned that it was not easy and he had to chase her, due to schedule conflicts.

 Welch accepted his invitation and the two recorded "Sweet Nothing".

 The track premiered on "The Chris Moyles Show" on BBC Radio 1 on 28 August 2012 and was later released as the fifth single from the album on 12 October 2012.
</item 9>

<item 10>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 10>
=== END ITEM id=7ccbd42f89d30432 ===

=== ITEM id=fa464bf72d443af6 ===
Gold sentences and facts:
Sentence 0:  The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made 1996 one of the most prolific years for material released by Prince.
  Fact 0.0: The album was Prince's third album released in 1996.
  Fact 0.1: One of Prince's other 1996 releases was "Chaos and Disorder".
  Fact 0.2: One of Prince's other 1996 releases was the soundtrack album of the Spike Lee movie "Girl 6".
  Fact 0.3: 1996 was one of the most prolific years for material released by Prince.
Sentence 1:  His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
  Fact 1.0: Prince's production company is 40 Acres and a Mule Filmworks.
  Fact 1.1: 40 Acres and a Mule Filmworks has produced over 35 films since 1983.

Retrieved items:
<item 1>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 1>

<item 2>
Emancipation (Prince album)
Emancipation is the nineteenth studio album by American recording artist Prince.

 It was released on November 19, 1996 by NPG Records and EMI Records.

 The title refers to Prince's freedom from his contract with Warner Bros.

 Records after 18 years, with which he had a contentious relationship.

 The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made
1996 one of the most prolific years for material released by Prince.
</item 2>

<item 3>
Cannibal boom
Only four other cannibal films were made after 1981 until the fad's conclusion in 1988 with Antonio Climati's "Natura Contro".
</item 3>

<item 4>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 4>

<item 5>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 5>

<item 6>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 6>

<item 7>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 7>

<item 8>
Theresa Randle
Theresa E. Randle (born December 27, 1964) is an American actress.

 She has appeared in films such as "Malcolm X" (1992), "Sugar Hill" (1994), "Beverly Hills Cop III" (1994), "Bad Boys" (1995), "Girl 6" (1996), "Space Jam"
(1996), "Spawn" (1997), and "Bad Boys II" (2003).
</item 8>

<item 9>
The Doom Generation
The Doom Generation is a 1995 American dark comedy film written and directed by Gregg Araki.

 It stars James Duval, Rose McGowan, and Johnathon Schaech.

 The film follows two troubled teenage lovers Amy Blue (McGowan) and Jordan White (Duval) who pick up a young handsome drifter named Xavier Red (Schaech).

 After he accidentally kills a store's clerk, the trio embarks on a journey full of sex, violence, and people from Amy's past.

 Billed as "A Heterosexual Movie by Gregg Araki", "The Doom Generation" is the second film in the director's trilogy known as the "Teenage Apocalypse Trilogy",
the first being "Totally Fucked Up" (1993) and the last one "Nowhere" (1997).

 The characters of Amy Blue and Jordan White are based on the Mark Beyer comic strip "Amy and Jordan".
</item 9>

<item 10>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 10>
=== END ITEM id=fa464bf72d443af6 ===

=== ITEM id=26a7282cfc218c0f ===
Gold sentences and facts:
Sentence 0: Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.
  Fact 0.0: Aleksandr Ivanovsky lived from 1881 to 1968.
  Fact 0.1: Aleksandr Ivanovsky was a Russian screenwriter.
  Fact 0.2: Aleksandr Ivanovsky was a Russian film director.
  Fact 0.3: Aleksandr Ivanovsky worked during the Soviet era.
Sentence 1: Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera" (1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".
  Fact 1.0: Arthur Lubin was born on July 25, 1898.
  Fact 1.1: Arthur Lubin died on May 12, 1995.
  Fact 1.2: Arthur Lubin was an American film director.
  Fact 1.3: Arthur Lubin was an American producer.
  Fact 1.4: Arthur Lubin directed several 'Abbott & Costello' films.
  Fact 1.5: Arthur Lubin directed 'Phantom of the Opera', released in 1943.
  Fact 1.6: Arthur Lubin directed the 'Francis the Talking Mule' series.
  Fact 1.7: Arthur Lubin created the talking-horse TV series 'Mister Ed'.

Retrieved items:
<item 1>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 1>

<item 2>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 2>

<item 3>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 3>

<item 4>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 4>

<item 5>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 5>

<item 6>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 6>

<item 7>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 7>

<item 8>
The entries profile Russian and Soviet figures across science, arts, politics, and activism: entomologist and prolific author Pavel Marikovsky; film director
Aleksandr Ivanovsky; animator Vladimir Danilevich; actors and collectors Ekaterina and Vladimir Semenikhin; and actor Aleksei Petrenko. They also describe the
anti-tsarist Petrashevsky Circle and member Nikolay Speshnev, plus Marxist revolutionaries, thinkers, and officials including Bazarov, Bag
</item 8>

<item 9>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 9>

<item 10>
Richard Lund
Richard Lund (9 July 1885 – 27 September 1960) was a Swedish film and theatre actor.

 He made his stage debut at Stora Teatern in Gothenburg in 1904 and later appeared in 73 films between 1912 and 1952, making his most important roles during the
silent film era.

 Among his best-known roles is that of Sir Archie in Mauritz Stiller's "Sir Arne's Treasure" (1919).

 He was the most prominent "first lover" of Swedish film during his heyday.
</item 10>
=== END ITEM id=26a7282cfc218c0f ===

=== ITEM id=5d37c4a0bc3d030a ===
Gold sentences and facts:
Sentence 0: The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.
  Fact 0.0: The 1892 Auburn Tigers football team represented Auburn University.
  Fact 0.1: The 1892 Auburn Tigers football team played in the 1892 college football season.
Sentence 1:  It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.
  Fact 1.0: The 1892 Auburn Tigers football team was the first college football team fielded by the Agricultural and Mechanical College of Alabama.
  Fact 1.1: The Agricultural and Mechanical College of Alabama is now known as Auburn University.
Sentence 2:  With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in Alabama.
  Fact 2.0: Auburn University has more than 22,000 undergraduate students.
  Fact 2.1: Auburn University has a total enrollment of more than 28,000.
  Fact 2.2: Auburn University has 1,260 faculty members.
  Fact 2.3: Auburn University is the second largest university in Alabama.

Retrieved items:
<item 1>
1983 Alabama Crimson Tide football team
The 1983 Alabama Crimson Tide football team (variously "Alabama", "UA", "Bama" or "The Tide") represented the University of Alabama in the 1983 NCAA Division
I-A football season.

 It was the Crimson Tide's 91st overall and 50th season as a member of the Southeastern Conference (SEC).

 The team was led by head coach Ray Perkins, in his first year, and played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in
Birmingham, Alabama.

 They finished the season with a record of eight wins and four losses (8–4 overall, 4–2 in the SEC).

 Ray Perkins, who played as a wide receiver for Bear Bryant in the 1960s, was named as the new head coach at Alabama on December 14, 1982, to succeed Bryant
after his 25-year tenure as Alabama's head coach.
</item 1>

<item 2>
1899 Connecticut Aggies football team
The 1899 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1899 college football season.

 This was the fourth year that the school fielded a football team.

 The Aggies were led by first year head coach T. D. Knowles, and completed the season with a record of 6–2.
</item 2>

<item 3>
1997 Alabama Crimson Tide football team
The 1997 Alabama Crimson Tide football team represented the University of Alabama in the college football season of 1997–98.

 The team's head coach was Mike DuBose, who was entering his first year at Alabama.

 They played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in Birmingham, Alabama and competed in the West Division of the
Southeastern Conference.

 Alabama would finish with a record of 4–7 record in suffering the program's first losing season since the 1984 season.

 The loss against Kentucky marked Alabama's first ever overtime game, as overtime rules for college football had gone into effect the previous season.
</item 3>

<item 4>
Auburn University
Auburn University (AU or Auburn) is a public research university in Auburn, Alabama, United States.

 With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in
Alabama.

 Auburn University is one of the state's two public flagship universities.
</item 4>

<item 5>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 5>

<item 6>
1906 Alabama Crimson White football team
The 1906 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1906 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 14th overall and 11th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach J. W. H. Pollard, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of five wins and one loss (5–1 overall, 3–1 in the SIAA).
</item 6>

<item 7>
1902 Connecticut Aggies football team
The 1902 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1902 college football season.

 This was the seventh year that the school fielded a football team.

 The Aggies were led by first year head coach Edwin O. Smith, and completed the season with a record of 4–3.
</item 7>

<item 8>
1905 Alabama Crimson White football team
The 1905 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1905 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 13th overall and 10th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach Jack Leavenworth, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of six wins and four losses (6–4 overall, 4–4 in the SIAA).
</item 8>

<item 9>
West Alabama Tigers football
The West Alabama Tigers football program is the intercollegiate American football team for the University of West Alabama located in the U.S. state of Alabama.

 The team competes in the NCAA Division II and are members of the Gulf South Conference.

 West Alabama's first football team was fielded in 1938.

 The team plays its home games at the 7,000 seat Tiger Stadium in Livingston, Alabama.

 The Tigers are coached by Brett Gilliland.
</item 9>

<item 10>
1892 Auburn Tigers football team
The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.

 It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.

 The squad was first coached by George Petrie.

 Auburn shut out Georgia Tech, 26–0, just two days after being shut out themselves by North Carolina, 64–0.

 The team finished the season with a record of 2–2.
</item 10>
=== END ITEM id=5d37c4a0bc3d030a ===

=== ITEM id=b1417a364935c179 ===
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
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 1>

<item 2>
Maria Katzarava
Maria Katzarava (born 1984) is an opera singer from Mexico who achieved international fame by winning first place in the Operalia competition in the opera and
zarzuela categories.

 Prior to this, she came to the attention of notable Mexican tenor Ramón Vargas, who helped her get a scholarship, and then won Mexico's Carlo Morelli national
singing competition in 2005, which earned her a debut at the Palacio de Bellas Artes.

 Since winning the Operalia competition, Katzarava has performed in Europe, where she currently lives, Asia and the Americas.

 She currently trains with soprano Mirella Freni.
</item 2>

<item 3>
Ao Li
Ao Li (born 8 February 1988) is a Chinese operatic bass-baritone and voice teacher who is particularly known for his performances at the San Francisco Opera.

 In 2013 he won first prize in the Operalia, The World Opera Competition and in 2014 he won the Metropolitan Opera National Council Auditions.
</item 3>

<item 4>
Guanqun Yu
Guanqun Yu (; born 1982) is a Chinese soprano who has sung in opera houses and concert halls internationally.

 In 2008 she won the Belvedere International Singing Competition and in 2012 she placed 2nd in the Operalia, The World Opera Competition.

 She is particularly known for portraying heroines in the operas of Wolfgang Amadeus Mozart, Giacomo Puccini, and Giuseppe Verdi.
</item 4>

<item 5>
Janet Williams (soprano)
Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera,
Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco
Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with
conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard,
Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
</item 5>

<item 6>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 6>

<item 7>
Rachel Willis-Sørensen
Rachel Willis-Sørensen (born 1984) is an American operatic soprano.

 She was a winner of the Metropolitan Opera National Council Auditions in 2010.

 In 2014, she won first prize, the Birgit Nilsson Prize, and the Zarzuela Prize of the Operalia competition.

 Other awards include first place in the 2009 Eleanor McCollum Competition for Young Singers in Houston and the 2010 Sara Tucker Grant from the Richard Tucker
Music Foundation.
</item 7>

<item 8>
Andrea Carroll (soprano)
Andrea Carroll is an American soprano who has had an active international career in concerts and operas since 2012.

 A finalist in the 2015 Operalia, The World Opera Competition, she is particularly known for her performances with the Houston Grand Opera and the Vienna State
Opera.
</item 8>

<item 9>
Norine Burgess
Norine Burgess is a Canadian singer.

 She is a graduate of the University of Calgary and the University of Toronto’s Opera School, mezzo-soprano.

 She received additional training as a member of the Canadian Opera Company (COC) Ensemble where she appeared in Electra, Suor Angelica, Lulu and Der
Rosenkavalier.

 Ms. Burgess also performed Le Nozze di Figaro(Cherubino), Ariadne auf Naxos (Dryad), La Traviata(Flora) and Die Zauberflöte (Second Lady) with the COC.

 Additional operatic successes include Le Nozze di Figaro (Cherubino) and Carmen (Mercedes) with the Vancouver Opera, Fenena in Nabucco with Manitoba Opera,
Albert Herring (Nancy) with the Calgary Opera and many appearances with the Edmonton Opera.
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
=== END ITEM id=b1417a364935c179 ===

=== ITEM id=875fb3159cb383fa ===
Gold sentences and facts:
Sentence 0: Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.
  Fact 0.0: Migene González-Wippler is a Puerto Rican.
  Fact 0.1: Migene González-Wippler is a new-age author.
  Fact 0.2: Migene González-Wippler is a leading expert on the Afro-Caribbean religion of Santería.
Sentence 1: Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among West African descendants.
  Fact 1.0: Santería is also known as Regla de Ochá.
  Fact 1.1: Santería is also known as La Regla de Ifá.
  Fact 1.2: Santería is also known as Lucumi.
  Fact 1.3: Santería is an Afro-American religion.
  Fact 1.4: Santería is a religion of Caribbean origin.
  Fact 1.5: Santería developed in the Spanish Empire.
  Fact 1.6: Santería developed among West African descendants.

Retrieved items:
<item 1>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 1>

<item 2>
Santería
Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among
West African descendants.

 "Santeria" is a Spanish word that means the "worship of saints".

 Santería is influenced by and syncretized with Roman Catholicism.

 Its sacred language, a variety of Yoruba, is the Lucumí language.
</item 2>

<item 3>
Ethnic religion
In religious studies, ethnic religion (or indigenous religion) is the expression of religion associated with a particular ethnic group.

 Ethnic religions are often distinguished from religions which claim to not be limited in ethnic or national scope, such as Christianity or Islam.

 Ethnic religions do not have to be excluded to independent religions.

 Some localized denominations of global religions are practiced solely by certain ethnic groups, with ethnic groups like the Assyrians following a unique
denominational structure of Christianity known as the Assyrian Church of the East.
</item 3>

<item 4>
Migene González-Wippler
Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.

 González-Wippler was born in Puerto Rico and has degrees in psychology and anthropology from the University of Puerto Rico and from Columbia University.

 In addition to her solid background in social sciences she has also worked as a science editor for the Interscience Division of John Wiley, the American
Institute of Physics, and the American Museum of Natural History, and as an English editor for the United Nations in Vienna, where she resided for many years.

 She is a cultural anthropologist and lectures frequently at universities and other educational institutions.

 She also has contributed extensively to the Institute of Puerto Rican Culture's collection, especially when it comes to Santeria and its practices, beliefs and
organization.
</item 4>

<item 5>
Ewe people
The Ewe people (Ewe: "Eʋeawó" , lit.

 "Ewe people"; or " Eʋedukɔ́ ", lit.

 "Ewe nation","Eʋenyigba" Eweland;) are an African ethnic group.

 They are the largest ethnic group in Togo (32%), the third largest ethnic group in Ghana (14%), and are a minority ethnic group in southern Benin, Ivory Coast
and Nigeria.

 They speak the Ewe language (Ewe: "Eʋegbe" ) which belongs to the Niger-Congo family of languages.

 They are related to other speakers of Gbe languages, such as, the Fon, Gen, Phla Phera, and the Aja people of Togo and Benin.
</item 5>

<item 6>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 6>

<item 7>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 7>

<item 8>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 8>

<item 9>
Awithlaknannai Mosona
It is thought that the Spanish had brought Alquerque to the American Southwest, and Awithlaknannai Mosona may have been an evolution from Alquerque.

 However, in Stewart Culin's 1907 book, the Zunis claim that they had adopted a hunt game from Mexico similar to Catch the Hare and the Fox games of Europe, and
transformed it into Awithlaknannai Mosona.

 In these games, one player has more pieces over the other, however, the other player's piece has more powers.

 The Zuni's equalized the numbers of pieces and their powers, and also may have transformed the board making its length far exceed its width.

 Diagonal lines also replaced orthogonal lines altogether.

 However, the hunt game from Mexico may have used an Alquerque board even though the game mechanics of their new game, Awithlaknannai Mosona, were completely
different.
</item 9>

<item 10>
Cora language
Cora is an indigenous language of Mexico of the Uto-Aztecan language family.

 It is spoken by the ethnic group that is widely known as the Cora but who refer to themselves as "Naáyarite".

 The Cora inhabit the northern sierra of the Mexican state Nayarit which is named after its indigenous inhabitants.

 Cora is a Mesoamerican language and shows many of the traits defining the Mesoamerican Linguistic Area.

 Under the General Law of Linguistic Rights of the Indigenous Peoples it is recognized as a "national language" along with 62 other indigenous languages and
Spanish which have the same "validity" in Mexico .
</item 10>
=== END ITEM id=875fb3159cb383fa ===

=== ITEM id=be6210ae9bf41241 ===
Gold sentences and facts:
Sentence 0:  He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.
  Fact 0.0: The person finished second in voting for the Heisman Trophy in 1956.
  Fact 0.1: Paul Hornung finished first in that Heisman Trophy voting in 1956.
Sentence 1: Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.
  Fact 1.0: Paul Vernon Hornung was born on December 23, 1935.
  Fact 1.1: Paul Vernon Hornung is nicknamed "The Golden Boy".
  Fact 1.2: Paul Vernon Hornung is a former professional American football player.
  Fact 1.3: Paul Vernon Hornung is a Hall of Fame running back.
  Fact 1.4: Paul Vernon Hornung played for the Green Bay Packers of the National Football League (NFL).
  Fact 1.5: Paul Vernon Hornung played for the Green Bay Packers from 1957 to 1966.
  Fact 1.6: Paul Vernon Hornung won four NFL titles.
  Fact 1.7: Paul Vernon Hornung won the first Super Bowl.

Retrieved items:
<item 1>
1997 Tennessee Volunteers football team
The 1997 Tennessee Volunteers football team represented the University of Tennessee during the 1997 NCAA Division I-A football season.

 Quarterback Peyton Manning had already completed his degree in three years, and had been projected to be the top overall pick in the 1997 NFL Draft, but
returned to Tennessee for his senior year.

 The Volunteers opened the season with victories against Texas Tech and UCLA, but for the third time in his career, Manning fell to Florida, 33–20.

 The Vols won the rest of their regular season games, finishing 10–1, and advanced to the SEC Championship Game against Auburn.

 Down 20–7, Manning led the Vols to a 30–29 victory.

 Throwing for four touchdowns, he was named the game's MVP, but injured himself in the process.

 The #3 Vols were matched up with #2 Nebraska in the Orange Bowl.

 Had Tennessee won and top-ranked Michigan lost to Washington State in the Rose Bowl, the Vols would have been expected to win the national championship.
</item 1>

<item 2>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 2>

<item 3>
1997 Tennessee Volunteers football team
However, the Vols' defense could not stop Nebraska's rushing attack, giving up more than 400 yards on the ground in a 42–17 loss.

 As a senior, Manning won numerous awards.

 He was a consensus first-team All-American and won the Maxwell Award, the Davey O'Brien Award, the Johnny Unitas Award, and the Best College Football Player
ESPY Award, among others.

 However, he did not win the Heisman Trophy, finishing runner-up to Charles Woodson, a CB from Michigan, and the only defensive player ever to win the Heisman
Trophy.
</item 3>

<item 4>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 4>

<item 5>
Frank Eliscu
Frank Eliscu (1912–1996) was an American sculptor and art teacher who designed and created the Heisman Memorial Football Trophy in 1935 when he was only 20
years old.

 The first Heisman Trophy, a strong young bull of a football player cast in bronze, was presented to a college football player in 1935, and is considered one of
the greatest honors a college athlete can receive.

 Over the years the Heisman Committee has paid tribute to Eliscu and his creation several times, and in 1985, the Heisman Committee invited him to speak at the
50th Anniversary Heisman ceremonies.
</item 5>

<item 6>
Terry Baker
Terry Wayne Baker (born May 5, 1941) is a former American football and basketball player.

 He played college football and college basketball at the Oregon State University.

 He played as a quarterback for the Oregon State Beavers football team from 1960 to 1962, winning the Heisman Trophy as senior.

 In the spring of his senior year, he played in the Final Four of the 1963 NCAA Men's Division I Basketball Tournament with the Oregon State Beavers men's
basketball team.

 To date, he is the only athlete to win a Heisman Trophy and play in the Final Four.

 Baker was the first overall pick in the 1963 NFL draft and played with the Los Angeles Rams of the National Football League (NFL) from 1963 to 1965.

 He then played for one season in the Canadian Football League (CFL) with the Edmonton Eskimos, in 1966.

 Baker was inducted into the College Football Hall of Fame in 1982.
</item 6>

<item 7>
Heisman Trophy
The Heisman Memorial Trophy (usually known colloquially as the Heisman Trophy or The Heisman), is awarded annually to the most outstanding player in college
football in the United States whose performance best exhibits the pursuit of excellence with integrity.

 Winners epitomize great ability combined with diligence, perseverance, and hard work.

 It is presented by the Heisman Trophy Trust in early December before the postseason bowl games.
</item 7>

<item 8>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 8>

<item 9>
1976 NCAA Division I football season
The 1976 NCAA Division I football season ended with a championship for the Panthers of the University of Pittsburgh.

 Led by head coach Johnny Majors (voted the AFCA Coach of the Year), the Pitt Panthers brought a college football championship to the home of the defending pro
football champions, the Steelers.

 Pitt also had the Heisman Trophy winner, Tony Dorsett; the Panthers had been ranked ninth in the preseason AP poll.
</item 9>

<item 10>
Johnny Majors
John Terrill Majors (born May 21, 1935) is a former American football player and coach.

 A standout halfback at the University of Tennessee, he was an All-American in 1956 and a two-time winner of the Southeastern Conference Most Valuable Player
award, in 1955 and 1956.

 He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.

 Majors served as the head football coach at Iowa State University (1968–1972), the University of Pittsburgh (1973–1976, 1993–1996), and Tennessee (1977–1992),
compiling a career college football record of 185–137–10.

 His 1976 Pittsburgh squad won a national championship after capping a 12–0 season with a victory in the Sugar Bowl.

 Majors was inducted into the College Football Hall of Fame as a player in 1987.
</item 10>
=== END ITEM id=be6210ae9bf41241 ===

=== ITEM id=a5e81235a0d0bb69 ===
Gold sentences and facts:
Sentence 0: "Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series overall.
  Fact 0.0: Moho House is the twenty-first episode of the twenty-eighth season of the animated television series The Simpsons.
  Fact 0.1: Moho House is the 617th episode of The Simpsons overall.
  Fact 0.2: The Simpsons is an animated television series.
Sentence 1:  On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
  Fact 1.0: Fox announced on May 4, 2015.
  Fact 1.1: Fox announced that The Simpsons had been renewed for season 28.

Retrieved items:
<item 1>
The Simpsons Theme
"The Simpsons" Theme", also referred to as "The Simpsons" Main Title Theme" in album releases, is the theme music of the animated television series "The
Simpsons".

 It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style
theme.

 The piece, which took 3 days, 2 hours, 48 minutes, and 19 seconds to create, has been noted by Elfman as the most popular of his career.

 The theme, as used for the opening sequence, was re-arranged during season 2, and the current arrangement by Alf Clausen was introduced at the beginning of the
third season.
</item 1>

<item 2>
List of The Simpsons comics
Since then, many more titles have been published, out of which "Simpsons Comics", "Bart Simpson", "Bart Simpson's Treehouse of Horror", "Simpsons Super
Spectacular", Simpsons Summer Shindig, and "Simpsons Winter Wingding".
</item 2>

<item 3>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 3>

<item 4>
Graph query row:
series: The Simpsons
series_kind: series
relationship: PART_OF
episode: Moho House
episode_kind: episode
</item 4>

<item 5>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 5>

<item 6>
List of The Simpsons comics
The following is a list of comic book series based on the animated TV show The Simpsons and published by Bongo Comics in the United States.

 The first comic strips based on "The Simpsons" appeared in 1991 in the magazine "Simpsons Illustrated" (not to be confused with the comic publications from
2012 bearing the same name), which was a companion magazine to the show.

 The comic strips were popular and a one-shot comic book entitled "Simpsons Comics and Stories", containing three different stories, was released in 1993 for
the fans.

 The book was a success and due to this, the creator of "The Simpsons", Matt Groening, and his companions Bill Morrison, Mike Rote, Steve Vance and Cindy Vance
created the publishing company Bongo Comics.

 By the end of 1993, Bongo was publishing four titles: "Simpsons Comics", "Bartman", "Radioactive Man" and "Itchy & Scratchy Comics".
</item 6>

<item 7>
The Simpsons (season 28)
The twenty-eighth season of the animated television series "The Simpsons" began airing on Fox in the United States on September 25, 2016, and ended on May 21,
2017.

 On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
</item 7>

<item 8>
The Simpsons Spin-Off Showcase
"The Simpsons Spin-Off Showcase" is the twenty-fourth episode of the eighth season of "The Simpsons".

 It originally aired on the Fox network in the United States on May 11, 1997.

 The episode centers on fictional pilot episodes of non-existent television series derived from "The Simpsons", and is a parody of the tendency of networks to
spin off characters from a hit series.

 As such it includes references to many different TV series.

 The first fictional spin-off is "Chief Wiggum P.I.", a cop-drama featuring Chief Wiggum and Seymour Skinner.

 The second is "The Love-matic Grampa", a sitcom featuring Moe Szyslak who receives dating advice from Abraham Simpson, whose ghost is possessing a love testing
machine.

 The final segment is "The Simpson Family Smile-Time Variety Hour", a variety show featuring the Simpson family except for Lisa, who has been replaced.
</item 8>

<item 9>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 9>

<item 10>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 10>
=== END ITEM id=a5e81235a0d0bb69 ===
