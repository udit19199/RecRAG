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

This file has 2 items.

=== ITEM id=a97063ccf5e585dc ===
Gold sentences and facts:
Sentence 0: The Bears' defense is all over the Vikings tonight, and Minnesota can't get out of its own way.
  Fact 0.0: The Bears' defense is all over the Vikings tonight.
  Fact 0.1: Minnesota cannot get out of its own way.
Sentence 1: Week 4 was the first time all year the Vikings DIDN'T throw the ball on at least 69% of their snaps.
  Fact 1.0: Week 4 was the first time all year the Vikings did not throw the ball on at least 69% of their snaps.
Sentence 2: 11:03 p.m. — The Vikings are using their two timeouts, but there isn't much they'll be able to do once this hits fourth down.
  Fact 2.0: At 11:03 p.m., the Vikings are using their two timeouts.
  Fact 2.1: There is not much the Vikings will be able to do once this hits fourth down.
Sentence 3: The new man on the Vikings, Josh Dobbs, keeps producing the same results they were enjoying offensively with Kirk Cousins, thanks to more help from the running game for Kevin O'Connell and the defense overall under Brian Flores.
  Fact 3.0: Josh Dobbs is the new man on the Vikings.
  Fact 3.1: Josh Dobbs keeps producing the same offensive results the Vikings enjoyed with Kirk Cousins.
  Fact 3.2: Josh Dobbs's results have more help from the running game under Kevin O'Connell.
  Fact 3.3: Josh Dobbs's results have more help from the defense overall under Brian Flores.

Retrieved items:
<item 1>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
This team is relatively young. They are scrappy. The fact that they have been there before gives them a chance. I think the quarterback is growing.”

• Cincinnati Bengals: Can Joe Burrow hold up for a full season?

What I think: Burrow is looking healthier over the past two weeks. The bye week comes at a great time for Burrow to get even healthier, but check out the
schedule on the other side: San Francisco, Buffalo, Houston, Baltimore, Pittsburgh and Jacksonville. Cincy still has a second game with the Steelers, plus games
against Kansas City and Cleveland. I’m worried Burrow will wear down against these defenses.

Exec comment: “You cannot always expect Joe Burrow to create magic with a bottom-half offensive line. The line is my concern all the way.”

Advertisement

2. Are teams sometimes going too far playing to win? The Eagles come to mind after falling 20-14 to the Jets.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Joe Burrow, Houston, Kansas City, Cincinnati Bengals, Eagles, Buffalo, Super Bowl, Jets, Cleveland, N.F.L., Baltimore, Pittsburgh,
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders, Jacksonville
</item 1>

<item 2>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
They are gaining more than 15 yards on just 11 percent of pass plays, down from 21 percent through six games last season and 19 percent for 2022 overall. Jalen
Hurts, A.J. Brown, DeVonta Smith and Dallas Goedert are still on the team, so I’m inclined to trust the production will return. But there’s also a lingering
fear defenses have caught up to aspects of the Eagles’ scheme. Does Philly have a counterpunch?

Advertisement

Exec comment: “People are going to crush Hurts and the offense. They are just churning the clock so much. It just feels like they are imposing their will on
teams through the run game and through their short passing and it just hasn’t taken off from an explosive passing standpoint, so when they play a team that can
move the ball on them through the air, it presents some challenges.”

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, The New York Times, Eagles, Jalen Hurts, DeVonta Smith, Super Bowl, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders, A.J.
Brown, Dallas Goedert
</item 2>

<item 3>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Jets would have taken possession with barely more than a minute remaining, no timeouts and Zach Wilson behind center facing a ferocious Eagles pass rush.
Was having Hurts pass in a situation when gaining a first down is unlikely the smart play under the circumstances?

“Are coaches not willing to go against the slight statistical benefit that WPA (win probability added) provides?” the exec asked. “Is it no longer OK to let
your defense win the game when it clearly has a huge talent/performance advantage? This Eagles game was 14-12, not 37-35.”

Critics attuned to the trend toward aggressiveness pounce when teams pass up chances to go for the kill. Coaches know team owners tend to be fans, not football
strategists. They know owners listen to these criticisms. That can provide an incentive to make the aggressive play, to go down swinging.

“Everybody thinks they are playing Brady, Manning or Rodgers in two-minute,” a veteran coach said.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Brady, Eagles, Hurts, Super Bowl, Zach Wilson, Jets, N.F.L., Manning, Rodgers, Eagles game, the exec, a veteran coach, WPA (win
probability added), two-minute
</item 3>

<item 4>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
Especially when you face a defensive coordinator who says, ‘You show me you can beat me throwing the ball outside the numbers — best of luck, and to do that,
you are going to have to take a five-step drop and I’m going to have Myles Garrett chasing you every play.'”

Advertisement

• Kansas City Chiefs: Is the explosive passing game gone for good?

What I think: The Chiefs have never scored less per game on offense through six games or a full season since Patrick Mahomes became their quarterback. They have
also never enjoyed larger average point differentials (9.8 per game), because their defense is so much better. I think the pass offense will become slightly
more explosive as Mahomes gets reps with his receivers, but I’m not expecting a huge shift.

The Chiefs are averaging 9.5 plays per touchdown drive when starting at least 70 yards from the opponent’s goal line, per TruMedia.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Patrick Mahomes, Super Bowl, N.F.L., TruMedia, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders, Myles
Garrett, Kansas City Chiefs
</item 4>

<item 5>
Graph query row:
article: Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game
published_on: 2023-11-30T22:10:58+00:00
publisher: CBSSports.com
mentioned_entities: [{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Brock Purdy'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Mark Andrews'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Alex Anzalone'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'NFL'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jordan Love'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Chris Olave'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Taysom Hill'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Alvin Kamara'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Jaguars'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name':
'New Orleans'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Vikings'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Russell Wilson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Saints'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Jayden Reed'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Puka Nacua'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Tyler Conklin'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Lockett'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'DAL'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'YAC'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Singletary'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'L.A.'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Shaheed'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Pete Werner'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Cameron Jordan'},
{'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'hand injury'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Falcons
defense'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Tim Tebow'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Tim
Hasselbeck'}, {'labels': ['__KGBuilder__', '__Entity__', 'Thing'], 'name': 'O-line'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Pickett'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Conner'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'PIT'}, {'labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'name': "Pittsburgh's unit"}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Rushing'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Titans DST'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Colts
DST'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jason Pierre-Paul'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Phillips'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Dolphins DST'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'Commanders DST'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Lavonte David'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'],
'name': 'Devin White'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jamel Dean'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'],
'name': 'Tampa Bay'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Godwin'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'LAR'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'L.A. Rams'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': '49ers DST'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Eagles DST'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'name': 'pass rush'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Lineup Decisions'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'],
'name': 'Pierce'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'A.J. Dillon'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Tank Dell'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Kyle Pitts'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Rhamondre Stevenson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Titans'}, {'labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'name': 'Bengals'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'Pass Rush Pressure Rate'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'name': 'Sacks'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Niners'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'name': 'Packers'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'blitz'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Sam Howell'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Brian Robinson'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Travis Kelce'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Patrick Mahomes'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'name': 'Cincinnati'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Cooper Kupp'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Curtis Samuel'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Rondale Moore'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'name': 'Texans'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Cardinals'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Josh Dobbs'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'C.J. Stroud'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Jaelan Phillips'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Dolphins'}, {'labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Eagles'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'CLE'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Demario Douglas'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Logan Thomas'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Deebo Samuel'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jalen Hurts'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'DeVonta Smith'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Justin Herbert'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Tua Tagovailoa'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Sam LaPorta'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Derek Carr'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Houston Texans'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Nico Collins'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Michael Pittman'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Tim Boyle'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Mike Evans'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Cooper'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'fantasy football'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Marquise Brown'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Greg Dortch'},
{'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'PPR'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Rashid Shaheed'}, {'labels':
['__KGBuilder__', '__Entity__', 'Topic'], 'name': 'ankle injury'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'DK Metcalf'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jaylen Waddle'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'DeAndre Hopkins'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Courtland Sutton'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Romeo Doubs'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Chris Godwin'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Ravens'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Diontae Johnson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Josh Downs'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'George Pickens'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Drake London'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jalen Guyton'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Quentin
Johnston'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Michael Gallup'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'James Conner'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Antonio Gibson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'Jets'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Garrett Wilson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'Chiefs'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Breece Hall'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Isiah Pacheco'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Lions'}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'], 'name':
'rib injury'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Brown'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Ferguson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Dak Prescott'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Amon-Ra St. Brown'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': "Ja'Marr Chase"}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'name': 'injuries'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': "De'Von Achane"}, {'labels': ['__KGBuilder__', '__Entity__', 'Topic'],
'name': 'concussion protocol'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jerry Jeudy'}, {'labels': ['__KGBuilder__', '__Entity__',
'Organization'], 'name': 'Browns'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Rams'}, {'labels': ['__KGBuilder__', '__Entity__',
'Person'], 'name': 'Dorian Thompson-Robinson'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Seattle Seahawks'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Geno Smith'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Seahawks'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Murray'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Tyreek Hill'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Austin Ekeler'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jahan Dotson'}, {'labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Commanders'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Terry McLaurin'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Pittsburgh'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Wilson'},
{'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Denver'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jared Goff'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Gardner Minshew'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'Jacksonville'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Joe Flacco'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Jahmyr Gibbs'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'David Montgomery'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'],
'name': 'Zack Moss'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Travis Etienne'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'],
'name': 'Raheem Mostert'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Bijan Robinson'}, {'labels': ['__KGBuilder__', '__Entity__',
'Person'], 'name': 'Rachaad White'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Tony Pollard'}, {'labels': ['__KGBuilder__', '__Entity__',
'Person'], 'name': 'Joe Mixon'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Javonte Williams'}, {'labels': ['__KGBuilder__', '__Entity__',
'Person'], 'name': 'Derrick Henry'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': "D'Andre Swift"}, {'labels': ['__KGBuilder__', '__Entity__',
'Person'], 'name': 'Jaylen Warren'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jerome Ford'}, {'labels': ['__KGBuilder__', '__Entity__',
'Person'], 'name': 'Najee Harris'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jaxon Smith-Njigba'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Tyler Lockett'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Kenny Pickett'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'name': 'Chargers'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Gerald Everett'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Metcalf'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'A.J. Brown'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'CeeDee Lamb'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Brandin Cooks'}, {'labels':
['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Cowboys'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Giants'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'DaRon Bland'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Stephon Gilmore'}, {'labels':
['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Kyler Murray'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Christian Watson'},
{'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Denver'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Buccaneers'},
{'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Baker Mayfield'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name':
'Patriots'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Broncos'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name':
'Hunter Henry'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Panthers'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'Colts'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Thomas Brown'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'],
'name': 'Arizona'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Chuba Hubbard'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'],
'name': 'Adam Thielen'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Bryce Young'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'],
'name': 'Zach Charbonnet'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Devin Singletary'}, {'labels': ['__KGBuilder__', '__Entity__',
'Person'], 'name': 'Christian McCaffrey'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Chase Young'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Brandon Aiyuk'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Keenan Allen'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Jonathan Taylor'}, {'labels': ['__KGBuilder__', '__Entity__', 'Place'], 'name': 'Seattle'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Place'], 'name': 'Dallas'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Matthew Stafford'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Tyler Allgeier'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Rashee Rice'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Topic'], 'name': 'tight end'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Trey McBride'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Dalton Schultz'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'David Njoku'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'George Kittle'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jake Ferguson'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Cade Otton'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Juwan Johnson'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Organization'], 'name': 'Falcons'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Flacco'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Josh Allen'}, {'labels': ['__KGBuilder__', '__Entity__', 'Person'], 'name': 'Jake Browning'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Will Levis'}, {'labels': ['__KGBuilder__', '__Entity__', 'Product'], 'name': 'SportsLine'}, {'labels': ['__KGBuilder__',
'__Entity__', 'Person'], 'name': 'Pat Freiermuth'}, {'labels': ['__KGBuilder__', '__Entity__', 'Organization'], 'name': 'Steelers'}]
connected_articles: []
</item 5>

<item 6>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
• Baltimore Ravens: Are the Ravens better off shifting to more of a pass-oriented offense?

What I think: Baltimore has never scored fewer points or amassed less offensive EPA through the first six games of a season with Lamar Jackson in the lineup.
The schedule and injuries have played roles, but with the Ravens shifting toward a more conventional pass offense, are the explosive quarterback runs mostly in
the past? Jackson has one rush longer than 20 yards (he’s had four or five by now previously). He’s gaining at least eight yards on 15 percent of carries, half
his previous rate. I’m not expecting a consistent precision passing game to suddenly materialize. Will this new offense be tougher to defend?

Advertisement

Exec comment: “The defense keeps them in it, but I don’t necessarily think this offensive change is going to make them any better when it counts. Lamar Jackson
does not appear as dynamic as a runner, but in this offense, he does not need to be.

Graph facts:
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- 2023 Kentucky online sports betting sites: Best legal sportsbooks, promos, bonuses, mobile apps, how to bet -MENTIONS-> Lamar Jackson
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Lamar Jackson
- NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Lamar Jackson
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Lamar Jackson

Entities: Lamar Jackson, The New York Times, Baltimore Ravens, Super Bowl, N.F.L., N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders
</item 6>

<item 7>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
These game plans were conservative enough to signal a potential in-season quarterback change. Denver faces Green Bay and Kansas City over the next two weeks
before heading into its Week 9 bye.

5. You play to win the game, but can you win too quickly? The Giants and Vikings are test cases.

The New York Giants and Minnesota Vikings exceeded expectations under first-year head coaches in 2022. Both reached the playoffs. Both are struggling in their
second seasons, facing difficult questions about the future. Would they have been better off enjoying only moderate success early? That could depend on how well
the football leadership has communicated with ownership.

Every head coach and general manager wants to win as much as possible, but the goal is to build for the long term, which can require time to do things with
sustainability in mind. Bad teams that win right away can reset expectations for fans and ownership, making it less practical to take one step backward in the
short term for a better shot at taking two steps forward down the line.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> New York Giants
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> New York Giants
- Moneyline Betting - What is the Moneyline and How Does it Work? -MENTIONS-> New York Giants
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> New York Giants
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> New York Giants

Entities: New York Giants, The New York Times, Kansas City, Minnesota Vikings, Super Bowl, N.F.L., N.F.L. Executives on the Biggest Questions Facing Super Bowl
Contenders, Denver, Green Bay, playoffs, Week 9 bye, football
</item 7>

<item 8>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
That figure was 7.8 in Mahomes’ first year as the starter and has climbed every season since. It’s a testament to the Chiefs’ adaptability, but also a
reflection of plummeting explosive play rates.

Exec comment: “Yeah, it looks harder for them on offense until Mahomes finds (Travis) Kelce on every major third down, every major two-minute play, every major
touchdown. Kansas City has the luxury of their best players being their toughest players. Chris Jones is a very tough player. Kelce is a very tough player.
Mahomes is a very tough player. You wonder why they win the games? Because the games are won by the toughest teams most of the time.”

• Philadelphia Eagles: Has Jalen Hurts regressed? Where are the explosive pass plays?

What I think: Explosive passing is down through most of the league, including for the Eagles.

Graph facts:
- NFL standings: Updated AFC, NFC playoff picture for Week 13 of 2023 season -MENTIONS-> Philadelphia Eagles
- Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral -MENTIONS-> Philadelphia Eagles
- Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE -MENTIONS-> Philadelphia Eagles
- NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 -MENTIONS-> Philadelphia Eagles
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> Philadelphia Eagles

Entities: Philadelphia Eagles, The New York Times, Travis Kelce, Mahomes, Jalen Hurts, Super Bowl, Chiefs, N.F.L., Chris Jones, explosive play rates, explosive
passing, N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders
</item 8>

<item 9>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
There isn’t a Josh Allen on the Giants or Vikings.

Minnesota in particular finds itself in an interesting spot, with Kirk Cousins approaching free agency and playing well enough, most likely, to keep the Vikings
from climbing too high in the draft order.

Advertisement

“They already gave a contract to (tight end) T.J. Hockenson, and they are going to give one to (receiver) Justin Jefferson,” an exec said. “Now, what? Are you
going into the rookie pool and taking the fourth-best quarterback in the draft? You can look like Atlanta with (Desmond) Ridder pretty quick.”

6. Two-minute drill: There were already many ways to win a game before the Dolphins invented another one.

The Dolphins were one of 223 teams since 2000 to be favored by at least 13.5 points in a regular-season or playoff game. Seventeen of those 223 teams fell
behind by 14 or more. Six of the 17 won anyway.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- Alcoholism, Ayahuasca and the Enlightenment of an N.F.L. Player -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Vikings, Minnesota, Atlanta, Dolphins, Desmond Ridder, Super Bowl, Justin Jefferson, N.F.L., N.F.L. Executives on the Biggest
Questions Facing Super Bowl Contenders, Kirk Cousins, quarterback, two-minute drill, Giants, draft
</item 9>

<item 10>
NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to bottom for Week 11 (Sporting News, 2023-11-13T21:54:00+00:00)
But the reeling defense and pass protection continue to be un-winnable assets.

24. Atlanta Falcons 4-5 (20)

The Falcons were forced into another QB change because of injury. They found a few more answers offensively but that position still made it an overall mess. The
defense also doesn't seem to be well prepared to different styles of offenses.

25. Green Bay Packers 3-6 (25)

The Packers saw some improved work from Jordan Love and his young weapons as they spread the ball well in the passing game, but the run defense woes and key
holes in coverage have all but doomed their playoff hopes.

26. Los Angeles Rams 3-6 (26)

The Rams are hoping to find a few more answers during a bye, as Sean McVay has seen his good coaching job earlier in the season go by the wayside with key
offensive injuries.

Graph facts:
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> Jordan Love
- Fantasy Football Week 13 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jordan Love
- NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game -MENTIONS-> Jordan Love
- FanDuel Picks Week 15: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Jordan Love
- Thursday Night Football FanDuel Picks: NFL DFS lineup advice for Week 4 Lions-Packers single-game tournaments -MENTIONS-> Jordan Love

Entities: Jordan Love, Green Bay Packers, Bills, Bengals, Texans, NFL power rankings: Lions, 49ers level up; Texans jump Bengals, Bills; Jets, Patriots sink to
bottom for Week 11, Atlanta Falcons, Los Angeles Rams, Jets, Lions, Sporting News, Patriots, 49ers, Sean McVay
</item 10>
=== END ITEM id=a97063ccf5e585dc ===

=== ITEM id=addc4adfe5338aed ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that 'all of that, all of it, was built on lies'.
Sentence 1: And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.
  Fact 1.0: Bankman-Fried left Jane Street.
  Fact 1.1: Jane Street is a high-frequency trading firm.
  Fact 1.2: Bankman-Fried, the former billionaire, got his start in finance at Jane Street after MIT.
  Fact 1.3: Bankman-Fried persuaded Yedidia to join him as a trader at Alameda.
  Fact 1.4: Bankman-Fried later persuaded Yedidia to join him as a developer at FTX.
Sentence 2: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.
  Fact 2.0: Bankman-Fried, known as SBF, has pleaded not guilty to some seven charges.
  Fact 2.1: The charges are of fraud and conspiracy.
Sentence 3: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 3.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence.
  Fact 3.1: The defense argued that the FTX founder Bankman-Fried acted in good faith.
  Fact 3.2: The defense argued that Bankman-Fried never meant to commit fraud or steal.
  Fact 3.3: The defense argued that Bankman-Fried basically got in over his head.

Retrieved items:
<item 1>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Bankman-Fried faces seven federal charges, including wire fraud, securities fraud and money laundering, that could put him in prison for the rest of his life.
Thus far, Bankman-Fried, 31, has remained mostly quiet in court intently listening to witnesses and at times writing notes to his attorneys. But as Wang
testified against him, Bankman-Fried looked visibly upset, shifting his gaze from his former friend to the ground, and at one point putting his head in his
hands.

Sam Bankman-Fried listens as Assistant U.S. Attorney Nicolas Roos questions Gary Wang during Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Wang, 30, was technology chief for FTX, which spiraled into bankruptcy in November.
</item 1>

<item 2>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Caroline Ellison, Bankman-Fried, Alameda Research, wire fraud, Nishad Singh, embezzlement, money laundering, conspiracy, political
donations, In the end, the FTX trial was about the friends screwed along the way, Joe Bankman, FTX trial, FTX, Yedidia
</item 2>

<item 3>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Whatever records the Southern District of New York has for the transactions will be shown.

“What conversations happened between him and his co-conspirators that are now cooperating against him?”

The second part is where all the drama is likely to come, says Christopher LaVigne, a litigation partner and co-chair of the cryptocurrency practice at the law
firm Withers. Prosecutors have to connect those transactions to Bankman-Fried, show that he knew what he was doing was wrong, and prove that he lied about it
anyway.

“What was he saying to his parents and his other advisors about this?” LaVigne says. “What conversations happened between him and his co-conspirators that are
now cooperating against him?”

To further establish intent, the government can use Bankman-Fried’s own words. The indictment calls Bankman-Fried’s tweets in November 2022 “false and
misleading.”

“We had him going out directly to the internet,” LaVigne says.

Graph facts:
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> cryptocurrency
- The people who ruined the internet -MENTIONS-> cryptocurrency
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> cryptocurrency
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> cryptocurrency
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> cryptocurrency

Entities: cryptocurrency, FTX trial, Southern District of New York, Christopher LaVigne, Withers, transactions, Bankman-Fried’s tweets, indictment, The FTX
trial is bigger than Sam Bankman-Fried, FTX, Sam Bankman-Fried, The Verge
</item 3>

<item 4>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Sam Bankman-Fried, the founder of bankrupt cryptocurrency exchange FTX, is seen during a hearing as a U.S judge revoked his bail, at a courthouse in New York,
U.S., August 11, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

$65 billion line of credit

For months, Bankman-Fried has known that Wang and Ellison, who were integral members of his personal and professional inner circles, had turned on him. Both
pleaded guilty in December and have since been cooperating with the U.S. attorney's office in Manhattan. Wang's testimony, which stretched into Friday, was
given under a cooperation agreement with the government. Ellison is expected to take the stand under a similar arrangement.

U.S.

Graph facts:
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Ellison
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Ellison
- In the end, the FTX trial was about the friends screwed along the way -MENTIONS-> Ellison
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Ellison
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Ellison

Entities: Ellison, New York, Jane Rosenberg, FTX, U.S. attorney's office, courthouse, pleading guilty, Wang's testimony, Sex, Signal messages, and sabotaging
FTX's code: SBF execs and Bahamas roommates tell all in court, Wang, hearing, Sam Bankman-Fried, Reuters, Manhattan, CNBC
</item 4>

<item 5>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Ellison is expected to take the stand under a similar arrangement.

U.S. District Judge Lewis Kaplan presides as Gary Wang testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

Born in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to the Massachusetts Institute of Technology to study math and computer
science. He worked at Google after college. Wang, who first met Bankman-Fried during high school at a summer camp, owned 10% of Alameda, while his boss owned
the other 90%. Wang told the court about the advantages that Alameda received by having code baked into FTX's software that allowed special access to the crypto
exchange.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, New York City, Lewis Kaplan, Alameda, Minnesota, cryptocurrency, Signal, Bahamas, Ellison, China, Jane Rosenberg, FTX, Sex, Signal
messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, FTX software, fraud trial of Sam Bankman-Fried
</item 5>

<item 6>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Damian Williams
- Damian Williams -SAID-> financial fraud
- Damian Williams -PARTICIPATED_IN-> Press conference
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto

Entities: Damian Williams, crypto, alleged crimes, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take
the stand, Fortune, FTX, the government, Sam Bankman-Fried, the trial, MIT, fraud
</item 6>

<item 7>
Graph query row:
result_type: connection
entity: Sam Bankman-Fried
relationship: SAID
related: Alameda Research
page: None
passage: None
</item 7>

<item 8>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Witness Adam Yedidia answers questions during Sam Bankman-Fried fraud trial over the collapse of FTX, the bankrupt cryptocurrency exchange, at Federal Court in
New York City, October 5, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

Two of Sam Bankman-Fried's former friends from MIT, who also worked at crypto exchange FTX while living with the company's founder in the Bahamas, took the
stand in a Manhattan courtroom this week to testify against their former classmate, confidant, and boss — a man who allegedly ran a crypto empire that defrauded
thousands of customers out of billions of dollars. Gary Wang, the lesser-known co-founder of FTX, was asked by Assistant U.S. Attorney Nicolas Roos on Thursday,
"Did you commit financial crimes while working at FTX?" "Yes," responded Wang.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Adam Yedidia, New York City, Bahamas, Jane Rosenberg, FTX, Nicolas Roos, Sam Bankman-Fried fraud trial, Federal Court in New York City,
financial crimes, Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court, Sam Bankman-Fried, Manhattan, Reuters,
CNBC
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.

Graph facts:
- Thane Rehn -PARTICIPATED_IN-> SBF Trial
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> Thane Rehn
- Thane Rehn -SAID-> SBF
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Thane Rehn
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> stealing customer funds

Entities: Thane Rehn, Larry David, stealing customer funds, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand, Alameda Research, Fortune, FTX, Sam Bankman-Fried, Tom Brady, government
</item 9>

<item 10>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Assistant United States Attorney Nicolas Roos questions Gary Wang during Sam Bankman-Fried's fraud trial over the collapse of FTX, the bankrupt cryptocurrency
exchange, at Federal Court in New York City, U.S., October 6, 2023, in this courtroom sketch. Jane Rosenberg | Reuters

An $8 billion bug

Adam Yedidia, who was the prosecution's second witness on Wednesday, continued his testimony on Thursday. Yedidia met Bankman-Fried in college at MIT, and the
pair remained close friends. Yedidia, assuming a robotic posture on the stand, worked out of FTX's Hong Kong office from January to October of 2021 and then in
the Bahamas until last year's collapse. In his testimony, he referred to a group Signal thread called "People of the House," referring to Bankman-Fried's $35
million penthouse, where many employees lived.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Gary Wang
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Gary Wang
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Gary Wang
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Gary Wang
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Gary Wang

Entities: Gary Wang, Adam Yedidia, New York City, Signal, Bahamas, Jane Rosenberg, FTX, Nicolas Roos, Sex, Signal messages, and sabotaging FTX's code: SBF execs
and Bahamas roommates tell all in court, Federal Court, People of the House, Sam Bankman-Fried's fraud trial, $35 million penthouse, Sam Bankman-Fried, Reuters
</item 10>
=== END ITEM id=addc4adfe5338aed ===
