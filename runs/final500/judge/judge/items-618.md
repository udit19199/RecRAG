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

=== ITEM id=163572679f5c325a ===
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
Finish What Ya Started
"Finish What Ya Started" is a song by Van Halen taken from their 1988 album "OU812".

 Despite the album being seemingly complete, Eddie Van Halen came up with the riff at 2 in the morning and went down to his then-neighbor Sammy Hagar to show
it.

 Hagar let Eddie in, and the two played guitars in his balcony until they had a completed song.

 Once Eddie left, Hagar decided to write the lyrics despite being late at night.

 The theme wound up being unfulfilled sex, summed up by Hagar as "blue balls.

 In the song, Eddie Van Halen recorded his guitar part on a Fender Stratocaster plugged direct into the studio mixing console.

 The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
</item 2>

<item 3>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 3>

<item 4>
Yeah! (Fuel song)
"Yeah!"

 is the first song released from Fuel's album "Puppet Strings".

 It is also the first song released featuring original lead singer Brett Scallions since Fuel's 2003 studio album "Natural Selection".
</item 4>

<item 5>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 5>

<item 6>
OU812 Tour
The OU812 Tour was a concert tour by hard rock band Van Halen.

 It was the second tour to include dates in Japan, and was the second with Sammy Hagar as vocalist (who had settled into the role more by this point).

 It followed the Monsters of Rock Tour 1988, which had formed the first part of the promotion for the "OU812" album.
</item 6>

<item 7>
Keisuke Kuwata (album)
Keisuke Kuwata is the eponymous studio album recorded by Keisuke Kuwata, a frontman of the Japanese rock band Southern All Stars.

 It was released by Taishita label under the Victor Entertainment in July 1988, shortly after the 10th anniversary of the band.

 Aside from the 1982 live recording album "Kamon Yuzo and Victor Wheels Live" released under the pseudonym Yūzō Kamon, it was first Kuwata's solo effort.
</item 7>

<item 8>
Hoarse &amp; Roaring
Hoarse & Roaring is the debut full-length album by neofolk band Parlor Hawk, released in 2010 on Northplatte Records.

 The album was produced by Joshua James and featured Neon Trees bassist Branden Campbell on each track (except "Saddest Song") as well as Fictionist member
Stuart Maxfield on the tracks "14 Years" and "Home".

 It was mixed by Todd Burke (Ben Harper, Jack Johnson) and mastered by Reuben Cohen (Edward Sharpe & The Magnetic Zeros, Bruno Mars).
</item 8>

<item 9>
OU812
OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist
Sammy Hagar.

 Van Halen started work on the album in September 1987 and completed it in April 1988, just one month before its release.
</item 9>

<item 10>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 10>
=== END ITEM id=163572679f5c325a ===

=== ITEM id=b856c3ad5e41d3f1 ===
Gold sentences and facts:
Sentence 0: The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.
  Fact 0.0: The Celluloid Closet is a 1995 American documentary film.
  Fact 0.1: The Celluloid Closet was directed by Rob Epstein and Jeffrey Friedman.
  Fact 0.2: The Celluloid Closet was written by Rob Epstein and Jeffrey Friedman.
Sentence 1: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.
  Fact 1.0: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening".
  Fact 1.1: The film Station to Station is made up of 62 one-minute films.

Retrieved items:
<item 1>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 1>

<item 2>
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 2>

<item 3>
Romance of Celluloid
The Romance of Celluloid is a 1937 short black and white documentary film, narrated by Frank Whitbeck, which goes behind the scenes to look at the manufacture
of film and the making of motion pictures.

 The film was the first of the studio's "Romance of Celluloid" series, which also included;
</item 3>

<item 4>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 4>

<item 5>
Graph query row:
film: Station to Station
release_date: 2014
relationship: None
connected_entity: None
</item 5>

<item 6>
The Boxer (Kele Okereke album)
The Boxer is the debut solo album by Kele Okereke, the lead singer of British indie rock band Bloc Party.

 Okereke released the album under the professional name of Kele on 21 June 2010.

 As promotion, he uploaded the songs "Rise" and "Walk Tall" to his personal website on 13 May.

 The first single from "The Boxer" was "Tenderoni", released on 14 June, and the second, "Everything You Wanted", on 16 August.

 The album's third single "On The Lam" was released on 25 October.
</item 6>

<item 7>
Station to Station (film)
Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.

 It was distributed internationally in 2015 following its release at the Sundance Film Festival.
</item 7>

<item 8>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 8>

<item 9>
Metro Station (band)
Metro Station is an American pop rock band that was formed in Los Angeles by Trace Cyrus and Mason Musso.

 In late 2006, the band signed a recording contract with Columbia Records and RED Ink Records.

 The band is best known for the Top 10 "Billboard" hit single "Shake It" from the group's self-titled debut album.

 In 2010, tension between Cyrus and Musso caused the band to go on hiatus.

 In 2011 the band returned, however it was announced that Cyrus was no longer a part of the group and Musso had purchased the rights to the name.

 An EP entitled "Middle of the Night" was released in 2013, which was led by the single "Every Time I Touch You".

 In 2014 Cyrus and a new drummer, Spencer Steffan came to the band and a new single entitled "Love & War" was released.

 In 2015, the band released a second full-length album titled "Savior".
</item 9>

<item 10>
Two Weeks Notice
Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.

 The film was written and directed by Marc Lawrence, who had previously scripted two other box-office hits for Bullock, "Miss Congeniality" and "Forces of
Nature".

 Although critical response was mixed, the film received a successful box office run, both in the United States and globally.
</item 10>
=== END ITEM id=b856c3ad5e41d3f1 ===

=== ITEM id=51bca44aec9e4974 ===
Gold sentences and facts:
Sentence 0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who successfully worked as the film director, the screenwriter, the art director and the animator.
  Fact 0.0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич) was born on 4 September 1924.
  Fact 0.1: Vladimir Petrovich Danilevich died on 9 October 2001.
  Fact 0.2: Vladimir Petrovich Danilevich was a well-known Soviet animator.
  Fact 0.3: Vladimir Petrovich Danilevich was a well-known Russian animator.
  Fact 0.4: Vladimir Petrovich Danilevich successfully worked as a film director.
  Fact 0.5: Vladimir Petrovich Danilevich successfully worked as a screenwriter.
  Fact 0.6: Vladimir Petrovich Danilevich successfully worked as an art director.
  Fact 0.7: Vladimir Petrovich Danilevich successfully worked as an animator.
Sentence 1: Wolfgang Becker (born 22 June 1954) is a German film director and writer.
  Fact 1.0: Wolfgang Becker was born on 22 June 1954.
  Fact 1.1: Wolfgang Becker is a German film director.
  Fact 1.2: Wolfgang Becker is a German writer.
Sentence 2:  He is best known to the international audience for his work "Good Bye Lenin!
  Fact 2.0: Wolfgang Becker is best known to the international audience for his work "Good Bye Lenin!".
Sentence 3: " (2003).
  Fact 3.0: "Good Bye Lenin!" is from 2003.

Retrieved items:
<item 1>
Graph query row:
person: Wolfgang Becker
relationship: CREATED_BY
role: None
creative_work: Good Bye Lenin!
</item 1>

<item 2>
Good Bye Lenin! (soundtrack)
Good Bye Lenin!

 is the original soundtrack album of the film of the same title starring Daniel Brühl and Katrin Sass.
</item 2>

<item 3>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 3>

<item 4>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 4>

<item 5>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 5>

<item 6>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 6>

<item 7>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 7>

<item 8>
Good Bye, Lenin!
Good Bye, Lenin!

 is a 2003 German tragicomedy film, directed by Wolfgang Becker.

 The cast includes Daniel Brühl, Katrin Saß, Chulpan Khamatova, and Maria Simon.

 Most scenes were shot at the Karl-Marx-Allee in Berlin and around Plattenbauten near Alexanderplatz.
</item 8>

<item 9>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 9>

<item 10>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 10>
=== END ITEM id=51bca44aec9e4974 ===

=== ITEM id=286a561c4e117185 ===
Gold sentences and facts:
Sentence 0:  He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).
  Fact 0.0: The player played three seasons for the Philadelphia Eagles.
  Fact 0.1: The player's Philadelphia Eagles seasons are listed as 1940.
  Fact 0.2: The player played for the Pittsburgh Steelers.
  Fact 0.3: The player's Pittsburgh Steelers seasons are listed as 1941-1942.
Sentence 1:  He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his departure from the Eagles.
  Fact 1.0: The player was drafted by the Philadelphia Eagles.
  Fact 1.1: The player was drafted in the second round of the 2008 NFL Draft.
  Fact 1.2: The player played for the Washington Redskins for three seasons.
  Fact 1.3: The player played for the Washington Redskins after leaving the Eagles.

Retrieved items:
<item 1>
Don Looney
John Don Looney (September 2, 1916 – April 5, 2015) was a professional American football end in the National Football League.

 He was born in Sulphur Springs, Texas.

 He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).

 He was the first receiver in NFL history to have over 100 yards receiving in each of his first two games, a feat which was not equaled until the 2008 NFL
season by another Eagles wide receiver, DeSean Jackson.

 At the time of his death, Looney was the second oldest living former NFL player.

 He was the father of NFL running back Joe Don Looney, who later died in a one-person motorcycle accident after his NFL career ended.
</item 1>

<item 2>
1986 Tampa Bay Buccaneers season
The 1986 Tampa Bay Buccaneers season began with the team trying to improve on an 2–14 season.

 This was one of the worst seasons in franchise history.

 There is some sentiment that the 1986 team was even worse than the winless team of 1976, and the 473 points conceded was not beaten by any NFL team until the
2001 Indianapolis Colts gave up 486.

 The Buccaneers selected Bo Jackson with the top pick in the draft, but were unable to convince him to join the team.

 Three weeks after the draft, Jackson signed a three-year baseball contract with the Kansas City Royals.

 Despite holding four of the first forty selections in the draft, and the presence of a great influx of fresh talent from defunct USFL teams, the Buccaneers
were unable to find any impact players in either the draft or free agency.

 They entered the season with a roster nearly identical to the previous season's 2–14 team.
</item 2>

<item 3>
Alec Gibson
Alec Raymond Gibson (born December 9, 1963 in Columbus, Ohio) is a former American football defensive end in the National Football League for the Washington
Redskins.

 Gibson played in three games for the Washington Redskins in 1987.

 The Washington Redskins went 3-0 during the strike.

 Defeating the St. Louis Cardinals, New York Giants, and Dallas Cowboys.

 Gibson caused Hall of Famer Tony Dorsett to fumble the football twice on Monday Night Football in Dallas the last replacement game played in 1987.

 He played college football at Ventura Junior College where he was an All-American defensive tackle and at the University of Illinois as a defensive end.

 In 1988, he played Arena football for the New York Knights.

 Gibson played 11 of 12 games before injuring his right knee (ACL) which put an end to his career.
</item 3>

<item 4>
DeSean Jackson
DeSean William Jackson (born December 1, 1986) is an American football wide receiver for the Tampa Bay Buccaneers of the National Football League (NFL).

 He played college football for the University of California, Berkeley, where he was recognized as a consensus All-American.

 He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his
departure from the Eagles.

 Jackson has been selected to the Pro Bowl three times, and was the first player selected to the Pro Bowl at two different positions in the same year when he
was named to the 2010 Pro Bowl as a wide receiver and return specialist.
</item 4>

<item 5>
Brooklyn Dodgers (NFL)
The Brooklyn Dodgers were an American football team that played in the National Football League from 1930 to 1943, and in 1944 as the Brooklyn Tigers.

 The team played its home games at Ebbets Field of the baseball National League's team, the Brooklyn Dodgers.

 In 1945, because of financial difficulties and the increasing scarcity of major league-level players because of the war-time defense requirements at the height
of World War II, the team was merged with the Boston Yanks and were known as the Yanks for that season.

 This old NFL franchise was not related to the earlier (second incarnation) American Football League II with a franchise that played as the Brooklyn Tigers for
the first half of the 1936 season before moving to Rochester, New York and playing as the Rochester Tigers.

 Another NFL team that played in the Brooklyn borough was the Brooklyn Lions (which became the Brooklyn Horsemen after merging with a team from an earlier first
incarnation AFL of the same name) in 1926.
</item 5>

<item 6>
Brooklyn Dodgers (NFL)
Later co-owner and partner Dan Topping (1912–1974), pulled the Tigers team out of the old NFL in 1946 and placed it in the newly established rival professional
league – the All-America Football Conference, which shortly lasted until 1949 until several stronger teams from the AAFC merged with and entered a reorganized
NFL in 1950.

 It lasted until 1970 with the NFL-AFL (third) merger following the establishment of the first "Super Bowl" inter-league national championship game three years
before with the old NFL champions playing the victors of the latest rival fourth incarnation of the American Football League IV, formed in 1960 (now the
American Football Conference (AFC).
</item 6>

<item 7>
Verlon Biggs
Verlon Marion Biggs (March 16, 1943 – June 7, 1994) was an American football defensive end in the American Football League and National Football League.

 He played for the New York Jets (AFL) in Super Bowl III, but felt he didn't receive enough credit for the Jets' playoff win against the Oakland Raiders in the
AFL Championship Game that launched them into the Super Bowl.

 He played well but sulked until 1970, demanded more money, and wound up signing with Vince Lombardi's Washington Redskins of the NFL.

 Always a dominating defensive end, Verlon solidified George Allen's defense (Allen replaced Lombardi upon his death in July 1970) and led the Redskins into
Super Bowl VII.

 His nickname with the Redskins was "dirty Biggs" because of his extremely physical style of play.
</item 7>

<item 8>
Miracle at the New Meadowlands
The Miracle at the New Meadowlands, also called "New Miracle at the New Meadowlands and "Miracle at the Meadowlands II" is the term used by sportscasters and
Philadelphia Eagles fans to refer to an improbable come-from-behind win by the Philadelphia Eagles over rival team the New York Giants at New Meadowlands
Stadium on December 19, 2010.

 The game was a crucial one in the context of the season, played between two divisional rivals in Week 15 of the 2010 NFL season.

 With just under eight minutes to play in the fourth quarter, the Eagles trailed the Giants by 21 points.

 They went on to score four unanswered touchdowns in the final seven minutes and 28 seconds of play, including a punt returned for a touchdown by DeSean Jackson
as time expired.

 Jackson became the first player in NFL history to win a game by scoring on a punt return as time expired.

 The win allowed the Eagles to progress to the 2010 NFL playoffs, where they lost to eventual Super Bowl champion Green Bay.
</item 8>

<item 9>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 9>

<item 10>
Dallas Texans (NFL)
The Dallas Texans played in the National Football League (NFL) for one season, 1952, with a record of 1–11.

 The team is considered one of the worst teams in NFL history, both on (lowest franchise winning percentage) and off the field.

 The team was based first in Dallas, then Hershey, Pennsylvania, and Akron, Ohio, during its only season.

 The Texans were the last NFL team to fold.

 Many players on the 1952 roster went to the new Baltimore Colts franchise in 1953.

 The American Football League (AFL) had a 1960 charter member named the Dallas Texans (who later became the Kansas City Chiefs), but the AFL Texans have no
relationship with the earlier NFL team.
</item 10>
=== END ITEM id=286a561c4e117185 ===

=== ITEM id=8832ce1ce35d2c95 ===
Gold sentences and facts:
Sentence 0: Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.
  Fact 0.0: Lake Louisvilla is a neighborhood.
  Fact 0.1: Lake Louisvilla is partially located in Louisville, Kentucky.
Sentence 1:  It is located between Westport Road in Louisville and KY 22 in Oldham County.
  Fact 1.0: Lake Louisvilla is located between Westport Road in Louisville.
  Fact 1.1: Lake Louisvilla is located between KY 22 in Oldham County.
Sentence 2: Oldham County is a county located in the commonwealth of Kentucky.
  Fact 2.0: Oldham County is a county.
  Fact 2.1: Oldham County is located in the commonwealth of Kentucky.
Sentence 3:  As of the 2010 census, the population was 60,316.
  Fact 3.0: As of the 2010 census, the population of Oldham County was 60,316.
Sentence 4:  Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.
  Fact 4.0: Oldham County was a prohibition or completely dry county until January 2005.
  Fact 4.1: Oldham County stopped being completely dry in January 2005 as the result of a 2004 'moist' vote.
  Fact 4.2: The 2004 'moist' vote permitted sales of alcohol in restaurants that seat at least 100 patrons.
  Fact 4.3: The 2004 'moist' vote permitted alcohol sales only in restaurants where 70% or more of total revenue is derived from sales of food.

Retrieved items:
<item 1>
Oldham County, Kentucky
Oldham County is a county located in the commonwealth of Kentucky.

 As of the 2010 census, the population was 60,316.

 Its county seat is La Grange.

 The county is named for Colonel William Oldham.

 Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants
that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.

 After a vote in late 2015; Oldham county has become a completely wet county.
</item 1>

<item 2>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 2>

<item 3>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 3>

<item 4>
Lake Worth, Florida
Lake Worth is a city in Palm Beach County, Florida, United States, which takes its name from the body of water along its eastern border, originally called "Lake
Worth", and now generally known as the Lake Worth Lagoon.

 The lake itself was named for General William J. Worth, who led U.S. forces during the last part of the Second Seminole War.

 s of 2010 , the population estimated by the U.S. Census Bureau was 34,910.

 The city is included in the Miami-Fort Lauderdale-Pompano Beach Metropolitan Area, which is home to approximately 5,563,857 people.
</item 4>

<item 5>
Lake Louisvilla, Louisville
Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.

 It is located between Westport Road in Louisville and KY 22 in Oldham County.

 Lake Louisvilla was developed in the 1920s as a summer resort for people living in the city of Louisville.

 The state of Kentucky drained the lake in the late 1980s due to safety concerns regarding the stability of a dam.
</item 5>

<item 6>
Wheeling Island
Wheeling Island is the most populated island in the Ohio River.

 It lies within the city of Wheeling in Ohio County, West Virginia, in the United States.

 The 2000 census showed a resident population of 3,142 people on the island, which has a land area of 1.514 km² (374 acres).

 Neville Island, PA is larger (3.4 km²) but has fewer people (1,232).
</item 6>

<item 7>
Casey County, Kentucky
Casey County is a county located in the U.S. Commonwealth of Kentucky.

 As of the 2010 census, the population was 15,955.

 Its county seat is Liberty.

 The county was formed in 1806 from the western part of Lincoln County and named for Colonel William Casey, a pioneer settler who moved his family to Kentucky
in 1779.

 It is the only Kentucky county entirely in the Knobs region.

 Casey County is home to annual Casey County Apple Festival, and is a prohibition or dry county.

 It is considered part of the Appalachian region of Kentucky.
</item 7>

<item 8>
Garrard County, Kentucky
Garrard County ( ;) is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 16,912.

 Its county seat is Lancaster.

 The county was formed in 1796 and was named for James Garrard, Governor of Kentucky from 1796 to 1804.

 It is a prohibition or dry county but Lancaster is wet.

 Lancaster was founded as a settlement of log cabins in 1776 at a springs that later provided a constant source of water to early pioneers.

 It is one of the oldest cities in the Commonwealth.

 Boonesborough, 25 miles to the east, was founded by Daniel Boone in 1775.

 Lexington, 28 miles to the north, was founded in 1775.

 Stanford, originally known as St. Asaph, is 10 miles south of Lancaster.

 It too was founded in 1775.

 The oldest permanent settlemenet in Kentucky, Harrodsburg, was founded in 1774 and is 18 miles to the west.

 The present day courthouse is one of the oldest courthouses in Kentucky in continuous use.
</item 8>

<item 9>
Kentucky County, Virginia
Kentucky County (then alternately spelled Kentucke County) was formed by the Commonwealth of Virginia by dividing Fincastle County into three new counties:
Kentucky, Washington, and Montgomery, effective December 31, 1776.

 During the three and one-half years of Kentucky County's existence, its seat of government was Harrodstown (then also known as Oldtown, later renamed
Harrodsburg).
</item 9>

<item 10>
Squire Creek Country Club
The Squire Creek Lodge features two floors offering a combined total of seven bedrooms.

 The Squire Creek Pool overlooks the golf course and is served by the Waterside Cafe.

 The Squire Creek Development offers six residential estates including the Squire Creek Estates, Timberland Estates, Fairway Estates, The Fairways, Fairway
Villas, and The Park Homes.

 Squire Creek Country Club is the title sponsor of the Louisiana Peach Festival in Ruston.

 Squire Creek hosted the 2005 Western Athletic Conference Golf Championships.
</item 10>
=== END ITEM id=8832ce1ce35d2c95 ===

=== ITEM id=b655584008f3f8f0 ===
Gold sentences and facts:
Sentence 0:  He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.
  Fact 0.0: The person described in this sentence was the brother of Alfred Jodl.
  Fact 0.1: Alfred Jodl was Chief of the Operations Staff of the OKW.
Sentence 1: Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
  Fact 1.0: Alfred Josef Ferdinand Jodl lived from 10 May 1890 to 16 October 1946.
  Fact 1.1: Alfred Jodl was a German general.
  Fact 1.2: Alfred Jodl was a war criminal during World War II.
  Fact 1.3: Alfred Jodl served as Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).

Retrieved items:
<item 1>
Hasso von Wedel (general)
Hasso von Wedel (20 November 1898 – 3 January 1961) was a German general who commanded the Wehrmacht Propaganda Troops during World War II.

 He was directly subordinate to the head of OKW Operations Staff ("Wehrmachtführungsstab, WFSt.")

, General Alfred Jodl.

 Wedel's Propaganda Department had control over the propaganda units and served to mediate between them and the Reich Propaganda Ministry of Joseph Goebbels.
</item 1>

<item 2>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 2>

<item 3>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 3>

<item 4>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 4>

<item 5>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 5>

<item 6>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 6>

<item 7>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 7>

<item 8>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 8>

<item 9>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 9>

<item 10>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 10>
=== END ITEM id=b655584008f3f8f0 ===

=== ITEM id=d46b6d73ada24870 ===
Gold sentences and facts:
Sentence 0: Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
  Fact 0.0: Ernst Willi Messerschmid is Prof. Dr.
  Fact 0.1: Ernst Willi Messerschmid was born on May 21, 1945.
  Fact 0.2: Ernst Willi Messerschmid is a German physicist.
  Fact 0.3: Ernst Willi Messerschmid is a former astronaut.
Sentence 1: Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
  Fact 1.0: Philippe Perrin was born on January 6, 1963.
  Fact 1.1: Philippe Perrin is a Colonel in the French Air Force.
  Fact 1.2: Philippe Perrin is a French test pilot.
  Fact 1.3: Philippe Perrin is a former CNES astronaut.
  Fact 1.4: Philippe Perrin is a former European Space Agency astronaut.

Retrieved items:
<item 1>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 1>

<item 2>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 2>

<item 3>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 3>

<item 4>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 4>

<item 5>
Philippe Perrin (artist)
Philippe Perrin, (La Tronche, 10 August 1964) is a French artist and contemporary photographer who lives and works in Paris.
</item 5>

<item 6>
Jean-Loup Puget
Jean-Loup Puget (born 7 March 1947) is a French astrophysicist.

 His current research interests lie in the Cosmic Microwave Background.

 Jean-Loup Puget and his collaborators reported the first identification of the Cosmic infrared background using COBE data.

 He is also, along with Alain Léger, credited with the origin of the hypothesis that the series of infrared lines observed in numerous astrophysical objects are
caused by emission from polycyclic aromatic hydrocarbons.

 He is currently principal investigator of the HFI module of the Planck space mission.
</item 6>

<item 7>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 7>

<item 8>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 8>

<item 9>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 9>

<item 10>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 10>
=== END ITEM id=d46b6d73ada24870 ===

=== ITEM id=67f99387a9b84299 ===
Gold sentences and facts:
Sentence 0: Dana Robins Ivey (born August 12, 1941) is an American actress.
  Fact 0.0: Dana Robins Ivey was born on August 12, 1941.
  Fact 0.1: Dana Robins Ivey is an American actress.
Sentence 1: Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.
  Fact 1.0: Two Weeks Notice is a 2002 film.
  Fact 1.1: Two Weeks Notice is an Australian-American film.
  Fact 1.2: Two Weeks Notice is a romantic comedy film.
  Fact 1.3: Two Weeks Notice stars Hugh Grant.
  Fact 1.4: Two Weeks Notice stars Sandra Bullock.

Retrieved items:
<item 1>
Heather Burns
Heather Burns (born April 7, 1975) is an American actress, known for her role as Miss Rhode Island in the 2000 film "Miss Congeniality".

 Her other film appearances include "You've Got Mail" (1998), "Two Weeks Notice" (2002) and "Bewitched" (2005).
</item 1>

<item 2>
Two Weeks Notice
Two Weeks Notice is a 2002 Australian-American romantic comedy film starring Hugh Grant and Sandra Bullock.

 The film was written and directed by Marc Lawrence, who had previously scripted two other box-office hits for Bullock, "Miss Congeniality" and "Forces of
Nature".

 Although critical response was mixed, the film received a successful box office run, both in the United States and globally.
</item 2>

<item 3>
Dana (surname)
Dana ( or ) as a surname may have several origins.

 In England, it came from "dann", the valley of a meadow, and it may mean the dweller of that valley.

 In Continental Europe, it probably came from "Dane" (or Danish, from Denmark).

 This surname is related to Danese in Italy and it can be found mostly in the Piedmont region.

 It may also be a modification of Huguenot French origin, probably a variant of "d'Aunay", of geographical origin.

 It may be also a Gaelic patronymic, since it is a common forename in Ireland.

 Dana is a relatively common surname in the US, ranking 7161 out of 88,799 in the 1990 U.S. Census.

 Dana is also the Persian word for wisdom.
</item 3>

<item 4>
Jaimee Foxworth
Jaimee Monae Foxworth (born December 17, 1979) is an American actress and model.

 She is best known for her role of Judy Winslow, the youngest daughter of Carl and Harriette Winslow on the ABC/CBS sitcom "Family Matters".

 She later transitioned to pornographic films using the name Crave, starring in several titles.
</item 4>

<item 5>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 5>

<item 6>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 6>

<item 7>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 7>

<item 8>
When You Can Fly
When You Can Fly is the second studio album by Canadian country music artist Victoria Banks and her first album with record label distribution, which earned
Banks a 2010 Canadian Indie Award nomination and six nominations at the 2009 Canadian Country Music Awards including Album of the Year, Single of the Year ("The
Wheel"), Songwriter of the Year ("The Wheel"), Female Artist of the Year and Rising Star.

 The album was self-produced by Banks, also earning her a nomination for Producer of the Year.

 It was released on April 14, 2009 by On Ramp Records/EMI Canada.

 The album's first single "The Wheel," earned a 2010 Canadian Radio Music Award nomination.

 Subsequent singles included "When You Can Fly," "Kiss Me, "This Old Halo" and "Don't Leave the Leavin'."

 Music videos for "The Wheel" and "When You Can Fly" received airplay on CMT Canada.
</item 8>

<item 9>
Two Weeks (The Office)
"Two Weeks" is the twenty-first episode of the fifth season of the television series "The Office", and the 93rd overall episode of the series.

 It originally aired on NBC in the United States on March 26, 2009.

 In this episode, Michael, who has given his two weeks' notice to Dunder Mifflin, tries to convince others in the office to quit and join him in starting a new
paper company.

 Meanwhile, Pam spends her day trying to put together the new photocopier and becomes frustrated with her job.
</item 9>

<item 10>
Dana Ivey
Dana Robins Ivey (born August 12, 1941) is an American actress.

 She is a five-time Tony Award nominee for her work on Broadway, and won the 1997 Drama Desk Award for Outstanding Featured Actress in a Play for her work in
both "Sex and Longing" and "The Last Night of Ballyhoo".

 Her film appearances include "The Color Purple" (1985), "Dirty Rotten Scoundrels" (1988), "The Addams Family" (1991), "Two Weeks Notice" (2002), "Rush Hour 3"
(2007) and "The Help" (2011).
</item 10>
=== END ITEM id=67f99387a9b84299 ===

=== ITEM id=0ef0627b16ca393b ===
Gold sentences and facts:
Sentence 0: John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).
  Fact 0.0: John H. Garabedian is an American radio personality.
  Fact 0.1: John H. Garabedian is an American disc jockey.
  Fact 0.2: John H. Garabedian was born on December 20, 1941.
  Fact 0.3: John H. Garabedian is age 75.
Sentence 1: Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
  Fact 1.0: Charles A. Agemian was born July 22, 1909.
  Fact 1.1: Charles A. Agemian's life ended March 30, 1996.
  Fact 1.2: Charles A. Agemian was an American banker.
  Fact 1.3: Charles A. Agemian took early retirement from Chase Manhattan Bank.
  Fact 1.4: Charles A. Agemian was executive vice president of operations at Chase Manhattan Bank.
  Fact 1.5: Charles A. Agemian became chairman of the Hackensack Trust Co.
  Fact 1.6: Charles A. Agemian became chief executive officer of the Hackensack Trust Co.
  Fact 1.7: The Hackensack Trust Co. was later renamed Garden State National Bank.

Retrieved items:
<item 1>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 1>

<item 2>
Open House Party
Open House Party (most of the time, shortened to OHP) is an American radio show hosted on Saturday and Sunday nights by Kannon, who also hosts the afternoon
drive slot on 102.9 Now.

 The show promotes itself as "the Biggest Party on the Planet".

 It focuses on playing contemporary hit radio (CHR) music, also known as Top 40.

 The show differentiates itself from most Top 40 because it plays a high rotation of remixes.

 Unlike most radio stations or programs, every song played on OHP is by a listener request, rather than computer systems automatically inserting songs that may
not even be popular anyways.

 The show started in 1987, and was hosted by John Garabedian from September 1987 to January 2017.

 Garabedian created the show, and hosted both Saturday and Sunday nights for a long period of time.

 Open House Party is broadcast on more than 50 stations around the world.
</item 2>

<item 3>
Don Sherwood (DJ)
Don Sherwood (September 7, 1925 – November 6, 1983) was an American radio personality.

 He was a San Francisco, California, disc jockey during the 1950s and 1960s.

 Billed as "The World's Greatest Disc Jockey," Sherwood spent most of his career hosting a 6-9 a.m. weekday program on KSFO in San Francisco (560 kHz, 5000
watts), which was then owned by the singing cowboy actor Gene Autry.
</item 3>

<item 4>
Lin Brehmer
Lin Brehmer (born August 19, 1954) is a disc jockey and radio personality at WXRT in Chicago, Illinois.

 Brehmer has been serving as the morning-drive slot disc jockey since 1991.
</item 4>

<item 5>
M. G. Kelly
Gary Sinclair (born 1952 in Ada, Oklahoma), known professionally as Michael Gary “M.G.” Kelly and Machine Gun Kelly, is an American actor, disc jockey and radio
personality.

 In addition to hosting several radio programs over the years, Kelly has held several acting roles as a disc jockey; also, he has served as an offstage
announcer on two game shows.
</item 5>

<item 6>
Radio personality
A radio personality (American English) or radio presenter (British English), commonly referred to as a "disc jockey" or "DJ" for short, is a person who has an
on-air position in radio broadcasting.

 A radio personality that hosts a radio show is also known as a radio host, and in India and Pakistan as a radio jockey.

 Radio personalities who introduce and play individual selections of recorded music are known as disc jockeys.

 The term has evolved to also describe a person who mixes a continuous flow of recorded music in real time.

 Broadcast radio personalities may include talk radio hosts, AM/FM radio show hosts, and satellite radio program hosts.

 Notable radio personalities include pop music radio hosts Martin Block, Alan Freed, Dick Clark, Delilah Luke, Ameen Sayani, Wolfman Jack, and Casey Kasem,
shock jocks such as Don Imus and Howard Stern, as well as sports talk hosts such as Mike Francesa and political talk hosts such as Rush Limbaugh.
</item 6>

<item 7>
XY TV
It launched in December 2003 and distribution was limited to select cable companies.

 As of May 31, 2006, XY.tv has ceased operations, and is no longer broadcasting.

 It was wholly owned by American radio personality John Garabedian.
</item 7>

<item 8>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 8>

<item 9>
Big Wilson
Big Wilson (born Malcolm John Wilson, Jr.; October 3, 1924 – October 5, 1989) was an American radio personality.

 He worked as a disc jockey at WNBC AM in New York City from the early 1960s until 1974 and moved to Miami in 1975 where he worked for WIOD and WCIX-TV.

 He was one of the last two hosts of the NBC network program Monitor.

 Prior to moving to NYC, Wilson and his wife Jody lived in a riverfront home in Rocky River, Ohio.

 He commuted to Cleveland where he was the leading radio personality for some years.

 Tim Conway and other well-known persons visited their home.
</item 9>

<item 10>
Howie Klein
Howie Klein (born 1950) is an American writer, concert promoter, disc jockey, music producer, record label founder, record label executive, progressive
political activist, and adjunct professor of music.

 He is perhaps best known for his role as President of Reprise Records from 1989 to 2001.

 He appears occasionally as himself in music-related film documentaries and has received accolades for his stance against censorship and for his advocacy of
free speech protection.
</item 10>
=== END ITEM id=0ef0627b16ca393b ===

=== ITEM id=2402f62ab3d6aaa4 ===
Gold sentences and facts:
Sentence 0:  Jan Kodeš and Margaret Court won the singles titles.
  Fact 0.0: Jan Kodeš won the singles titles.
  Fact 0.1: Margaret Court won the singles titles.
Sentence 1:  She is currently a Christian minister in Perth, Western Australia.
  Fact 1.0: A woman referred to in this sentence is currently a Christian minister.
  Fact 1.1: The Christian minister referred to in this sentence is based in Perth, Western Australia.

Retrieved items:
<item 1>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 1>

<item 2>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 2>

<item 3>
1972 French Open – Men's Singles
Jan Kodes was the two-time defending champion, but lost in the quarterfinals to Patrick Proisy.
</item 3>

<item 4>
Graph query row:
entity: Margaret Court
relationship: LOCATED_IN
related: Perth
</item 4>

<item 5>
Margaret Court
Margaret Court {'1': ", '2': ", '3': ", '4': "} (née Smith; born 16 July 1942), also known as Margaret Smith Court, is a retired Australian tennis player and
former world No. 1.

 She is currently a Christian minister in Perth, Western Australia.

 In tennis, she amassed more major titles than any other player in history.
</item 5>

<item 6>
Martin Legner
Martin Legner (born December 17, 1961 in Tyrol) is a professional Austrian wheelchair tennis player who has been ranked number one for wheelchair doubles.

 Legner has won the Australian Open doubles title four times with Robin Ammerlaan since 2000, and has won eight doubles titles.

 Six of them are Australian Open Titles, the other two being French Open titles.

 He has won one French Open and one Australian Open singles title.

 He has represented his country at every Summer Paralympics since 1992, and has competed in both singles and doubles at all of those Games.

 His favorite surfaces are clay and hard court.

 As of April 7, 2007 his highest singles rank was number three.

 Also of that date, he was ranked number six for singles and doubles.

 His wheelchair is manufactured by Kronbickler.

 Legner is a fan of Bob Marley, and listens to his music aside from tennis.
</item 6>

<item 7>
1970 French Open
The 1970 French Open was a tennis tournament that was held at the Stade Roland Garros in Paris in France from 25 May through 7 June 1970.

 It was the 69th edition of the French Open, the 40th to be open to foreign competitors, and the second Grand Slam of the year.

 Jan Kodeš and Margaret Court won the singles titles.
</item 7>

<item 8>
Carlos Fernandes (tennis)
Carlos Fernandes (1936-) is a retired Brazilian tennis player.

 He had a good all-round game with excellent passing shots and moved fast around the court.

 Fernandes had a reputation for being a charismatic ladies man.

 He later became a coach.

 He made his Grand Slam debut at Wimbledon 1957, losing in the opening round to Jorgen Ulrich.

 At Roland Garros in 1958, Fernandes lost in round one to Pierre Darmon.

 At Wimbledon he lost in round one to Butch Buchholz.

 At 1959 French Open, Fernandes lost in round three to Jacques Brichant.At Wimbledon he lost in round two to Neale Fraser.

 At French Open 1960, Fernandes lost in round three to Bobby Wilson and at Wimbledon lost in round two to Wolfgang Stuck.

 At 1961 French Open, Fernandes achieved his best Grand Slam singles result by beating Pierre Darmon and Bob Hewitt before losing to Jan-Erik Lundquist in the
quarter finals.

 He lost in round two of Wimbledon to Wilson.

 At Roland Garros 1962, Fernandes lost to Lundquist in round two.
</item 8>

<item 9>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 9>

<item 10>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 10>
=== END ITEM id=2402f62ab3d6aaa4 ===

=== ITEM id=82d40131ad650b20 ===
Gold sentences and facts:
Sentence 0:  He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.
  Fact 0.0: The writer has written three novels.
  Fact 0.1: The writer has written three collections of short stories.
  Fact 0.2: The writer has written a collection of novellas.
  Fact 0.3: The writer has written seven volumes of essays.
  Fact 0.4: Three of the writer's essay volumes have won National Book Critics Circle Award prizes.
  Fact 0.5: One of the writer's essay volumes, 'A Temple of Texts' (2006), won the Truman Capote Award for Literary Criticism.
Sentence 1: Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for Literature.
  Fact 1.0: Naguib Mahfouz was born on December 11, 1911.
  Fact 1.1: Naguib Mahfouz died on August 30, 2006.
  Fact 1.2: Naguib Mahfouz's Arabic name is نجيب محفوظ, transliterated Nagīb Maḥfūẓ.
  Fact 1.3: Naguib Mahfouz was an Egyptian writer.
  Fact 1.4: Naguib Mahfouz won the 1988 Nobel Prize for Literature.

Retrieved items:
<item 1>
Graph query row:
person: Naguib Mahfouz
Nobel_Prize_entity: Nobel Prize in Literature
relationship: AWARDED
year: 1988
years: None
connected: True
</item 1>

<item 2>
Naguib Mahfouz Medal for Literature
The Naguib Mahfouz Medal for Literature is a literary award for Arabic literature.

 It is given to the best contemporary novel written in Arabic, but not available in English translation.

 The winning book is then translated into English, and published by American University in Cairo Press.

 It was first awarded in 1996 and is presented annually on December 11, the birthday of Nobel laureate Naguib Mahfouz, by the President of the American
University in Cairo.
</item 2>

<item 3>
Nobel Prize in Literature
Since 1901, the Nobel Prize in Literature (Swedish: "Nobelpriset i litteratur" ) has been awarded annually to an author from any country who has, in the words
of the will of Alfred Nobel, produced "in the field of literature the most outstanding work in an ideal direction" (original Swedish: "den som inom litteraturen
har producerat det mest framstående verket i en idealisk riktning").

 Though individual works are sometimes cited as being particularly noteworthy, here "work" refers to an author's work as a whole.

 The Swedish Academy decides who, if anyone, will receive the prize in any given year.

 The academy announces the name of the chosen laureate in early October.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895; the others are the Nobel Prize in Chemistry, Nobel Prize in Physics, Nobel
Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 3>

<item 4>
Frances Liardet
Frances E. Liardet is a writer and translator of Arabic literature.

 She studied Creative Writing at the University of East Anglia.

 She has translated several book-length works, including two books by the modernist Egyptian writer Edwar al-Kharrat and one by Nobel Prize winner Naguib
Mahfouz.
</item 4>

<item 5>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 5>

<item 6>
Naguib Mahfouz
Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for
Literature.

 He is regarded as one of the first contemporary writers of Arabic literature, along with Tawfiq el-Hakim, to explore themes of existentialism.

 He published 34 novels, over 350 short stories, dozens of movie scripts, and five plays over a 70-year career.

 Many of his works have been made into Egyptian and foreign films.
</item 6>

<item 7>
Thebes at War
Thebes at War is an early novel by the Egyptian writer Naguib Mahfouz.

 It was originally published in Arabic in 1944.

 An English translation by Humphrey Davies appeared in 2003.

 The novel is one of several that Mahfouz wrote at the beginning of his career, with Pharaonic Egypt as their setting.

 Others in this series of novels include "Khufu's Wisdom" (1939) and "Rhadopis of Nubia" (1943).

 All have been translated into English and appeared in one volume under the title "Three Novels of Ancient Egypt" (Everyman's Library, 2007).
</item 7>

<item 8>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 8>

<item 9>
Cairo Trilogy
The Cairo Trilogy (Arabic: الثلاثية‎ ‎ ("The Trilogy") or ثلاثية القاهرة ("The Cairo Trilogy")) is a trilogy of novels written by the Egyptian novelist and
Nobel Prize winner Naguib Mahfouz, and one of the prime works of his literary career.
</item 9>

<item 10>
The Beginning and the End (novel)
The Beginning and the End (بداية ونهاية) is a novel by Naguib Mahfouz, who was awarded the Nobel Prize in literature in 1988.

 An Egyptian, Naguib has been credited with modernizing Arabic literature, with his prolific writing style and his themes on existentialism.

 The novel is marked by very bold characterization by Naguib Mahfouz, for the time period that the story is set in.

 The story moves at a prolific pace and tries to look at the world from each characters view point.
</item 10>
=== END ITEM id=82d40131ad650b20 ===

=== ITEM id=dcf5b64813af4e94 ===
Gold sentences and facts:
Sentence 0: Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
  Fact 0.0: Dorian Gray is the main character of the novel The Picture of Dorian Gray.
  Fact 0.1: The Picture of Dorian Gray was published in 1890.
  Fact 0.2: The Picture of Dorian Gray was written by Oscar Wilde.
Sentence 1: The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".
  Fact 1.0: The Picture of Dorian Gray is a philosophical novel.
  Fact 1.1: The Picture of Dorian Gray was written by Oscar Wilde.
  Fact 1.2: The Picture of Dorian Gray was first published complete.
  Fact 1.3: The Picture of Dorian Gray was first published complete in the July 1890 issue of Lippincott's Monthly Magazine.
Sentence 2:  The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.
  Fact 2.0: The editor of Lippincott's Monthly Magazine feared the story was indecent.
  Fact 2.1: The editor of Lippincott's Monthly Magazine, without Wilde's knowledge, deleted roughly five hundred words before publication.

Retrieved items:
<item 1>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 1>

<item 2>
Dorian Gray (disambiguation)
Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
</item 2>

<item 3>
The Picture of Dorian Gray (1945 film)
The Picture of Dorian Gray is a 1945 American horror-drama film based on Oscar Wilde's 1890 novel of the same name.

 Released in March 1945 by Metro-Goldwyn-Mayer, the film is directed by Albert Lewin and stars George Sanders as Lord Henry Wotton and Hurd Hatfield as Dorian
Gray.

 Shot primarily in black-and-white, the film features four inserts in 3-strip Technicolor of Dorian's portrait as a special effect (the first two of his
portrait's original state, and the second two after a major period of degeneration).
</item 3>

<item 4>
Dorian Gray (1970 film)
Dorian Gray (Italian: "Il dio chiamato Dorian") aka "The Sins of Dorian Gray" is a 1970 movie adaptation of Oscar Wilde's novel "The Picture of Dorian Gray"
starring Helmut Berger.

 The Italian title translates as "A God Called Dorian".
</item 4>

<item 5>
The entries survey literature, theatre, film, music, and curiosities: French Decadence favored artifice and excess; Camus staged Dostoyevsky’s *The Possessed*
in 1959; Algren won the 1950 National Book Award; Poe satirized Gothic fiction; Moorcock wrote Elric and science-fiction novels; Nin’s diaries and erotica
appeared partly posthumously. Wilde’s *Dorian Gray* faced censorship and condemnation. Verne’s
</item 5>

<item 6>
The Picture of Dorian Gray (1917 film)
The Picture of Dorian Gray (German: Das Bildnis des Dorian Gray) is a 1917 German silent fantasy film directed by Richard Oswald and starring Bernd Aldor, Ernst
Pittschau, and Ernst Ludwig.

 The film is based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 6>

<item 7>
Dorian Gray (character)
Dorian Gray is the main character in Oscar Wilde's novel The Picture of Dorian Gray.

 Dorian Gray is a handsome, susceptible, wealthy young Victorian gentleman.

 He falls in love with and devotes himself to Sibyl Vane.

 Dorian has two friends, Basil and Henry.

 Basil is an artist who draws a portrait of Dorian that emphasizes his youthful, noble, and natural beauty.

 Henry is a friend who has a cynical nature and who shares various thoughts with Dorian.
</item 7>

<item 8>
The Picture of Dorian Gray (1910 film)
The Picture of Dorian Gray (1910), a.k.a. Dorian Grays Portræt, is a Danish silent film based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 8>

<item 9>
A Portrait of Dorian Gray
A Portrait of Dorian Gray (2005) is the fashion designer and photographer Karl Lagerfeld's rendition of Oscar Wilde's novel "The Picture of Dorian Gray" (1891)
in photography.

 The models, Larry Scott and Eva Herzigová, star as Mr. and Mrs. Dorian Gray.
</item 9>

<item 10>
Oscar Wilde’s philosophical novel *The Picture of Dorian Gray* first appeared in 1890; its magazine editor secretly cut about 500 words. Reviewers condemned it,
and Wilde defended the work before revising it for book publication. Its handsome, wealthy Victorian hero loves Sibyl Vane and is influenced by artist Basil and
cynical Lord Henry. Adaptations include Danish (1910) and German (1917) silent films, a 1945 MGM film starring Hurd Hatfield
</item 10>
=== END ITEM id=dcf5b64813af4e94 ===
